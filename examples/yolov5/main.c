/*
 * High-level flow
 * ---------------
 * - Main thread captures frames, detects the laser dot, and runs servo control.
 * - Worker thread consumes the latest frame and runs YOLO inference.
 * - Main thread uses the latest available cat track plus engagement-aware play
 *   mode selection (implemented in play_algorithms.cpp) to choose the next
 *   target point each control tick.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <signal.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <awnn_lib.h>

#include "tracking_utils.h"
#include "servo_control.h"
#include "play_algorithms.h"
#include "image_utils.h"
#include "yolov5_pre_process.h"
#include "yolov5_post_process.h"

struct InferenceShared {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    cv::Mat latest_frame;
    int has_new_frame;
    int stop;
    int inference_running;
    int has_cat_info;
    Yolov5CatTrackInfo latest_track;
    Yolov5CatDetections latest_detections;
};

struct InferenceThreadArgs {
    Awnn_Context_t *context;
    const char *frame_file;
    struct InferenceShared *shared;
};

static volatile sig_atomic_t g_sigint_received = 0;

static void handle_sigint(int signum) {
    (void)signum;
    g_sigint_received = 1;
}




static void print_pwm_sysfs_overview(void) {
    DIR *dir = opendir("/sys/class/pwm");
    if (dir == NULL) {
        fprintf(stderr, "PWM debug: /sys/class/pwm is unavailable on this system.\n");
        return;
    }

    fprintf(stderr, "PWM debug: discovered pwmchips in /sys/class/pwm:\n");
    struct dirent *entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "pwmchip", 7) != 0) {
            continue;
        }

        char npwm_path[256];
        snprintf(npwm_path, sizeof(npwm_path), "/sys/class/pwm/%s/npwm", entry->d_name);
        FILE *f = fopen(npwm_path, "r");
        int npwm = -1;
        if (f != NULL) {
            if (fscanf(f, "%d", &npwm) != 1) {
                npwm = -1;
            }
            fclose(f);
        }

        fprintf(stderr, "  - %s (npwm=%d)\n", entry->d_name, npwm);
    }
    closedir(dir);

    fprintf(stderr,
            "PWM debug: verify overlays are loaded and map PAN/TILT to valid channel indices.\n"
            "           example overlays: sun60iw2p1-pwm1-1 and sun60iw2p1-pwm1-2.\n");
}


struct RandomScanState {
    float target_pan_deg;
    float target_tilt_deg;
    float speed_deg_per_sec;
    int frames_until_retarget;
};

static const float RANDOM_MIN_SPEED_DEG_PER_SEC = 8.0f;
static const float RANDOM_MAX_SPEED_DEG_PER_SEC = 30.0f;

static float random_float_range(float min_v, float max_v) {
    const float r = (float)rand() / (float)RAND_MAX;
    return min_v + (max_v - min_v) * r;
}

static void retarget_random_scan(struct RandomScanState *scan_state) {
    scan_state->target_pan_deg = random_float_range(-45.0f, 45.0f);
    scan_state->target_tilt_deg = random_float_range(-30.0f, 30.0f);
    scan_state->speed_deg_per_sec = random_float_range(
        RANDOM_MIN_SPEED_DEG_PER_SEC,
        RANDOM_MAX_SPEED_DEG_PER_SEC);
    scan_state->frames_until_retarget = 25 + (rand() % 70);
}



static void apply_confidence_aware_servo_smoothing(
    ServoState *servo_state,
    const ServoState *pre_update_state,
    float confidence) {
    // Lower confidence -> heavier damping to reduce jitter.
    // Higher confidence -> more responsive movement.
    const float conf_norm = clampf((confidence - 0.25f) / 0.65f, 0.0f, 1.0f);
    const float alpha = 0.22f + 0.68f * conf_norm;

    servo_state->pan_deg = pre_update_state->pan_deg + alpha * (servo_state->pan_deg - pre_update_state->pan_deg);
    servo_state->tilt_deg = pre_update_state->tilt_deg + alpha * (servo_state->tilt_deg - pre_update_state->tilt_deg);
}

static float compute_laser_intensity_scale(
    const struct CatPlayState *play_state,
    const char *algo_name,
    float session_time_sec) {
    float intensity = 0.25f + 0.75f * clampf(play_state->engagement_score, 0.0f, 1.0f);

    if (play_state->director_intent == DIRECTOR_INTENT_CHASE) {
        intensity += 0.20f;
    } else if (play_state->director_intent == DIRECTOR_INTENT_TEASE) {
        intensity -= 0.08f;
    } else if (play_state->director_intent == DIRECTOR_INTENT_RECOVER) {
        intensity -= 0.12f;
    }

    if (algo_name != NULL &&
        (strcmp(algo_name, "hesitation_pause") == 0 || strcmp(algo_name, "near_miss_tease_pause") == 0)) {
        intensity *= 0.60f;
    }

    // Subtle heartbeat pulse before/inside pounce windows to build anticipation.
    if (play_state->director_intent == DIRECTOR_INTENT_POUNCE_WINDOW) {
        const float phase = 2.0f * 3.1415926f * 1.8f * session_time_sec;
        const float pulse = sinf(phase);
        const float heartbeat = 0.88f + 0.12f * pulse * pulse;
        intensity *= heartbeat;
    }

    // Tiny flicker at chase peaks (kept subtle to avoid harsh strobing).
    if (play_state->director_intent == DIRECTOR_INTENT_CHASE && intensity > 0.78f) {
        intensity += random_float_range(-0.05f, 0.05f);
    }

    return clampf(intensity, 0.08f, 1.0f);
}

static void apply_intensity_motion_style(
    ServoState *servo_state,
    const ServoState *pre_update_state,
    float intensity_scale,
    const struct CatPlayState *play_state) {
    float style_alpha = 0.30f + 0.70f * clampf(intensity_scale, 0.0f, 1.0f);
    if (play_state->director_intent == DIRECTOR_INTENT_CHASE) {
        style_alpha = clampf(style_alpha + 0.08f, 0.0f, 1.0f);
    }
    if (play_state->director_intent == DIRECTOR_INTENT_TEASE ||
        play_state->director_intent == DIRECTOR_INTENT_RECOVER) {
        style_alpha = clampf(style_alpha - 0.12f, 0.0f, 1.0f);
    }

    servo_state->pan_deg = pre_update_state->pan_deg + style_alpha * (servo_state->pan_deg - pre_update_state->pan_deg);
    servo_state->tilt_deg = pre_update_state->tilt_deg + style_alpha * (servo_state->tilt_deg - pre_update_state->tilt_deg);
}

static void update_random_scan_servo(
    ServoState *servo_state,
    struct RandomScanState *scan_state,
    float dt_sec) {
    if (scan_state->frames_until_retarget <= 0) {
        retarget_random_scan(scan_state);
    }

    float dx = scan_state->target_pan_deg - servo_state->pan_deg;
    float dy = scan_state->target_tilt_deg - servo_state->tilt_deg;
    float distance = sqrtf(dx * dx + dy * dy);
    const float max_step = scan_state->speed_deg_per_sec * dt_sec;

    if (distance < 0.8f) {
        retarget_random_scan(scan_state);
    } else if (distance <= max_step || max_step <= 0.001f) {
        servo_state->pan_deg = scan_state->target_pan_deg;
        servo_state->tilt_deg = scan_state->target_tilt_deg;
        scan_state->frames_until_retarget--;
    } else {
        const float scale = max_step / distance;
        servo_state->pan_deg += dx * scale;
        servo_state->tilt_deg += dy * scale;
        scan_state->frames_until_retarget--;
    }

    servo_state->pan_deg = clampf(servo_state->pan_deg, -45.0f, 45.0f);
    servo_state->tilt_deg = clampf(servo_state->tilt_deg, -45.0f, 45.0f);
}


static void probe_servo_signs_of_life(struct ServoPwm *pan_pwm,
                                      struct ServoPwm *tilt_pwm) {
    fprintf(stderr,
            "Servo probe: testing original mappings with visible pan/tilt motion...\n");

    // Pan sweep around center while tilt is held neutral.
    const float pan_probe_points[] = {-20.0f, 20.0f, 0.0f};
    for (unsigned int i = 0; i < sizeof(pan_probe_points) / sizeof(pan_probe_points[0]); ++i) {
        servo_pwm_set_angle(pan_pwm, pan_probe_points[i]);
        servo_pwm_set_angle(tilt_pwm, 0.0f);
        usleep(220000);
    }

    // Tilt sweep around center while pan is held neutral.
    const float tilt_probe_points[] = {-15.0f, 15.0f, 0.0f};
    for (unsigned int i = 0; i < sizeof(tilt_probe_points) / sizeof(tilt_probe_points[0]); ++i) {
        servo_pwm_set_angle(pan_pwm, 0.0f);
        servo_pwm_set_angle(tilt_pwm, tilt_probe_points[i]);
        usleep(220000);
    }

    fprintf(stderr,
            "Servo probe complete. If movement looked swapped, invert pan/tilt channels in main.c.\n");
}

static void run_laser_alignment_sequence(struct ServoPwm *pan_pwm,
                                         struct ServoPwm *tilt_pwm,
                                         struct MosfetPowerGpio *laser_gpio) {
    // Alignment routine for physical setup verification:
    // 1) Draw horizontal center line with laser ON.
    // 2) Turn laser OFF while repositioning to vertical start.
    // 3) Draw vertical center line with laser ON.
    const int steps = 26;

    mosfet_gpio_set(laser_gpio, true);
    for (int i = 0; i <= steps; ++i) {
        float t = (float)i / (float)steps;
        float pan = -35.0f + 70.0f * t;
        servo_pwm_set_angle(pan_pwm, pan);
        servo_pwm_set_angle(tilt_pwm, 0.0f);
        usleep(35000);
    }

    mosfet_gpio_set(laser_gpio, false);
    servo_pwm_set_angle(pan_pwm, 0.0f);
    servo_pwm_set_angle(tilt_pwm, -30.0f);
    usleep(180000);

    mosfet_gpio_set(laser_gpio, true);
    for (int i = 0; i <= steps; ++i) {
        float t = (float)i / (float)steps;
        float tilt = -30.0f + 60.0f * t;
        servo_pwm_set_angle(pan_pwm, 0.0f);
        servo_pwm_set_angle(tilt_pwm, tilt);
        usleep(35000);
    }
}

static void *inference_thread_main(void *arg) {
    struct InferenceThreadArgs *args = (struct InferenceThreadArgs *)arg;

    while (1) {
        cv::Mat frame;

        pthread_mutex_lock(&args->shared->mutex);
        while (!args->shared->has_new_frame && !args->shared->stop) {
            pthread_cond_wait(&args->shared->cond, &args->shared->mutex);
        }

        if (args->shared->stop) {
            pthread_mutex_unlock(&args->shared->mutex);
            break;
        }

        frame = args->shared->latest_frame.clone();
        args->shared->has_new_frame = 0;
        args->shared->inference_running = 1;
        pthread_mutex_unlock(&args->shared->mutex);

        Yolov5CatTrackInfo track_info = {0, 0, 0, 0, 0, 0};
        Yolov5CatDetections detections = {0};

        if (!frame.empty() && cv::imwrite(args->frame_file, frame)) {
            unsigned int file_size = 0;
            unsigned char *plant_data = yolov5_pre_process(args->frame_file, &file_size);
            if (plant_data != NULL) {
                void *input_buffers[] = {plant_data};
                awnn_set_input_buffers(args->context, input_buffers);
                awnn_run(args->context);
                float **results = awnn_get_output_buffers(args->context);
                yolov5_post_process(args->frame_file, results, &track_info, &detections);
                free(plant_data);
            }
        }

        pthread_mutex_lock(&args->shared->mutex);
        args->shared->latest_track = track_info;
        args->shared->latest_detections = detections;
        args->shared->has_cat_info = 1;
        args->shared->inference_running = 0;
        pthread_mutex_unlock(&args->shared->mutex);
    }

    return NULL;
}

static int parse_brightness_percent(const char *arg, unsigned int *brightness_percent) {
    if (arg == NULL || brightness_percent == NULL) {
        return -1;
    }

    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0' || parsed > 100UL) {
        return -1;
    }

    *brightness_percent = (unsigned int)parsed;
    return 0;
}

int main(int argc, char **argv) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

    if (argc < 2 || argc > 4) {
        fprintf(stderr,
                "Usage: %s <nbg> [camera_device] [laser_brightness_percent]\n"
                "  nbg: path to YOLOv5 .nb model\n"
                "  camera_device: optional V4L2 node (default: /dev/video0)\n"
                "  laser_brightness_percent: optional integer 0..100 (default: 100)\n",
                argv[0]);
        return -1;
    }

    const char *nbg = argv[1];
    const char *camera_device = "/dev/video0";
    unsigned int laser_brightness_percent = 100;

    if (argc >= 3) {
        // Backward compatible: allow ./yolov5 <nbg> <brightness> or <camera>.
        if (parse_brightness_percent(argv[2], &laser_brightness_percent) != 0) {
            camera_device = argv[2];
        }
    }
    if (argc >= 4 && parse_brightness_percent(argv[3], &laser_brightness_percent) != 0) {
        fprintf(stderr,
                "Invalid laser_brightness_percent '%s'. Expected integer 0..100.\n",
                argv[3]);
        return -1;
    }

    const unsigned int laser_pwm_cycle_ticks = 20;
    unsigned int laser_pwm_tick = 0;
    const unsigned int laser_pwm_on_ticks =
        (laser_brightness_percent * laser_pwm_cycle_ticks) / 100;

    fprintf(stderr,
            "Runtime config: camera=%s laser_brightness=%u%%\n",
            camera_device, laser_brightness_percent);
    const char *inference_frame_file = "live_frame.jpg";

    const unsigned int pan_pwm_chip = 10;
    const unsigned int pan_pwm_channel = 1;
    const unsigned int tilt_pwm_chip = 10;
    const unsigned int tilt_pwm_channel = 2;

    const char *mosfet_gpiochip_path = "/dev/gpiochip0";
    const unsigned int pan_power_gpio_line = 32;
    const unsigned int tilt_power_gpio_line = 33;

    // Dedicated laser control GPIO (A7Z pin 31: PB3 => gpiochip0 line 35).
    // Similar to Arduino laser pin control, but via Linux gpiochip line.
    const unsigned int laser_gpio_line = 35;

    const int input_channels = 3;
    const float control_dt_sec = 0.03f;
    const char *play_tuning_json_path = "examples/yolov5/play_tuning.json";

    srand((unsigned int)time(NULL));

    if (load_cat_play_tuning_json(play_tuning_json_path) == 0) {
        fprintf(stderr, "Loaded play tuning config: %s\n", play_tuning_json_path);
    } else {
        fprintf(stderr, "Play tuning config not loaded (%s); using built-in defaults.\n", play_tuning_json_path);
    }

    cv::VideoCapture camera(camera_device, cv::CAP_V4L2);
    if (!camera.isOpened()) {
        fprintf(stderr, "Failed to open webcam device: %s\n", camera_device);
        return -1;
    }

    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);
    if (context == NULL) {
        fprintf(stderr, "Failed to create NPU context with nbg: %s\n", nbg);
        camera.release();
        awnn_uninit();
        return -1;
    }

    struct MosfetPowerGpio pan_power_gpio = {0};
    struct MosfetPowerGpio tilt_power_gpio = {0};
    struct MosfetPowerGpio laser_gpio = {0};
    if (mosfet_gpio_open(&pan_power_gpio, mosfet_gpiochip_path, pan_power_gpio_line) < 0 ||
        mosfet_gpio_open(&tilt_power_gpio, mosfet_gpiochip_path, tilt_power_gpio_line) < 0 ||
        mosfet_gpio_open(&laser_gpio, mosfet_gpiochip_path, laser_gpio_line) < 0) {
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }
    fprintf(stderr,
            "GPIO mapping: pan_power=%s line %u (A7Z pin 29 / PB0), "
            "tilt_power=%s line %u (A7Z pin 30 / PB1), "
            "laser=%s line %u (A7Z pin 31 / PB3).\n",
            mosfet_gpiochip_path, pan_power_gpio_line,
            mosfet_gpiochip_path, tilt_power_gpio_line,
            mosfet_gpiochip_path, laser_gpio_line);
    if (mosfet_gpio_set(&pan_power_gpio, true) < 0 ||
        mosfet_gpio_set(&tilt_power_gpio, true) < 0 ||
        mosfet_gpio_set(&laser_gpio, true) < 0) {
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        mosfet_gpio_close(&laser_gpio);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }

    struct ServoPwm pan_pwm = {0};
    struct ServoPwm tilt_pwm = {0};
    if (servo_pwm_open(&pan_pwm, pan_pwm_chip, pan_pwm_channel) < 0 ||
        servo_pwm_open(&tilt_pwm, tilt_pwm_chip, tilt_pwm_channel) < 0 ||
        servo_pwm_set_angle(&pan_pwm, 0.0f) < 0 ||
        servo_pwm_set_angle(&tilt_pwm, 0.0f) < 0 ||
        servo_pwm_enable(&pan_pwm) < 0 ||
        servo_pwm_enable(&tilt_pwm) < 0) {
        fprintf(stderr,
                "Servo init failed for configured PAN(chip=%u,channel=%u) TILT(chip=%u,channel=%u).\n",
                pan_pwm_chip, pan_pwm_channel, tilt_pwm_chip, tilt_pwm_channel);
        print_pwm_sysfs_overview();
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        mosfet_gpio_close(&laser_gpio);
        servo_pwm_close(&pan_pwm);
        servo_pwm_close(&tilt_pwm);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }

    // Bare-minimum startup: center servos while laser stays enabled during runtime.
    if (servo_pwm_set_angle(&pan_pwm, 0.0f) < 0 ||
        servo_pwm_set_angle(&tilt_pwm, 0.0f) < 0) {
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        mosfet_gpio_close(&laser_gpio);
        servo_pwm_close(&pan_pwm);
        servo_pwm_close(&tilt_pwm);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }
    usleep(150000);

    probe_servo_signs_of_life(&pan_pwm, &tilt_pwm);

    struct InferenceShared inference_shared;
    pthread_mutex_init(&inference_shared.mutex, NULL);
    pthread_cond_init(&inference_shared.cond, NULL);
    inference_shared.has_new_frame = 0;
    inference_shared.stop = 0;
    inference_shared.inference_running = 0;
    inference_shared.has_cat_info = 0;
    inference_shared.latest_track.has_cat = 0;

    struct InferenceThreadArgs worker_args = {context, inference_frame_file, &inference_shared};
    pthread_t inference_thread;
    if (pthread_create(&inference_thread, NULL, inference_thread_main, &worker_args) != 0) {
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        mosfet_gpio_close(&laser_gpio);
        servo_pwm_close(&pan_pwm);
        servo_pwm_close(&tilt_pwm);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        pthread_mutex_destroy(&inference_shared.mutex);
        pthread_cond_destroy(&inference_shared.cond);
        return -1;
    }

    ServoState servo_state = {0.0f, 0.0f};
    cv::Point2f virtual_laser_point(0.0f, 0.0f);
    struct CatPlayState play_state = {0};
    init_cat_play_state(&play_state);
    int frame_index = 0;
    float play_session_time_sec = 0.0f;
    CatTrackFilterState track_filter = {0};
    MultiCatTrackerState multi_cat_tracker = {0};
    init_multi_cat_tracker_state(&multi_cat_tracker);

    // Deadman: if camera stream stalls, center servos and cut power.
    time_t last_frame_time = time(NULL);
    int servo_rails_powered = 1;
    int deadman_active = 0;

    int printed_resolution = 0;
    while (!g_sigint_received) {
        cv::Mat raw_frame;
        if (!camera.read(raw_frame) || raw_frame.empty()) {
            if (difftime(time(NULL), last_frame_time) > 2.0 && !deadman_active) {
                // Safety: no fresh camera for >2s, stop motion and power rails.
                servo_pwm_set_angle(&pan_pwm, 0.0f);
                servo_pwm_set_angle(&tilt_pwm, 0.0f);
                mosfet_gpio_set(&pan_power_gpio, false);
                mosfet_gpio_set(&tilt_power_gpio, false);
                mosfet_gpio_set(&laser_gpio, false);
                servo_rails_powered = 0;
                deadman_active = 1;
                fprintf(stderr, "Deadman engaged: camera stalled, servo rails powered off.\n");
            }
            usleep(100000);
            continue;
        }

        if (deadman_active && !servo_rails_powered) {
            if (mosfet_gpio_set(&pan_power_gpio, true) < 0 ||
                mosfet_gpio_set(&tilt_power_gpio, true) < 0) {
                fprintf(stderr, "Deadman recovery failed: unable to re-enable servo rails.\n");
                usleep(100000);
                continue;
            }
            // Keep laser ON during normal runtime after deadman recovery.
            mosfet_gpio_set(&laser_gpio, true);
            servo_rails_powered = 1;
            deadman_active = 0;
            fprintf(stderr, "Deadman cleared: camera recovered, servo rails re-enabled.\n");
        }

        if (!printed_resolution) {
            printed_resolution = 1;
        }

        cv::Mat frame = raw_frame;
        if (frame.channels() == 4) cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        else if (frame.channels() == 1) cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        if (frame.channels() != input_channels) continue;

        last_frame_time = time(NULL);

        pthread_mutex_lock(&inference_shared.mutex);
        inference_shared.latest_frame = frame.clone();
        inference_shared.has_new_frame = 1;
        pthread_cond_signal(&inference_shared.cond);

        Yolov5CatTrackInfo raw_track = inference_shared.latest_track;
        Yolov5CatDetections raw_detections = inference_shared.latest_detections;
        int has_track_info = inference_shared.has_cat_info;
        int inference_running = inference_shared.inference_running;
        pthread_mutex_unlock(&inference_shared.mutex);

        Yolov5CatTrackInfo active_track = {0, 0, 0, 0, 0, 0};
        if (has_track_info) {
            active_track = update_multi_cat_tracker_and_get_active(&multi_cat_tracker, &raw_detections);
        }
        if (!active_track.has_cat && has_track_info) {
            active_track = raw_track;
        }
        Yolov5CatTrackInfo smoothed = filter_cat_track(&track_filter, (active_track.has_cat ? &active_track : NULL));

        if (smoothed.has_cat) {
            cv::Point2f frame_center((float)frame.cols * 0.5f, (float)frame.rows * 0.5f);
            if (frame_index == 0) {
                virtual_laser_point = frame_center;
            }

            const char *algo_name = "unknown";
            cv::Point2f play_target = build_cat_play_target(
                &play_state,
                smoothed,
                smoothed.confidence,
                virtual_laser_point,
                frame_index,
                frame.cols,
                frame.rows,
                control_dt_sec,
                &algo_name);

            ServoState pre_update_state = servo_state;
            update_servo_state(&servo_state, frame_center, play_target, frame.cols, frame.rows);
            apply_confidence_aware_servo_smoothing(&servo_state, &pre_update_state, smoothed.confidence);
            const float intensity_scale = compute_laser_intensity_scale(&play_state, algo_name, play_session_time_sec);
            apply_intensity_motion_style(&servo_state, &pre_update_state, intensity_scale, &play_state);
            servo_pwm_set_angle(&pan_pwm, servo_state.pan_deg);
            servo_pwm_set_angle(&tilt_pwm, servo_state.tilt_deg);
            virtual_laser_point = play_target;
            play_session_time_sec += control_dt_sec;

            // Engagement-driven software PWM for brightness while a cat is active.
            const unsigned int effective_on_ticks = (unsigned int)(
                clampf((float)laser_pwm_on_ticks * intensity_scale, 0.0f, (float)laser_pwm_cycle_ticks) + 0.5f);
            const int laser_on_this_tick = (effective_on_ticks > 0) &&
                (laser_pwm_tick < effective_on_ticks);
            mosfet_gpio_set(&laser_gpio, laser_on_this_tick);
            laser_pwm_tick = (laser_pwm_tick + 1) % laser_pwm_cycle_ticks;

            fprintf(stderr,
                    "cat_conf=%.2f algo=%s engage=%.2f intensity=%.2f target=(%.1f,%.1f) servo=(%.2f,%.2f)\n",
                    smoothed.confidence, algo_name, play_state.engagement_score, intensity_scale, play_target.x, play_target.y,
                    servo_state.pan_deg, servo_state.tilt_deg);
        } else {
            servo_state.pan_deg = 0.0f;
            servo_state.tilt_deg = 0.0f;
            servo_pwm_set_angle(&pan_pwm, servo_state.pan_deg);
            servo_pwm_set_angle(&tilt_pwm, servo_state.tilt_deg);
            mosfet_gpio_set(&laser_gpio, true);
            virtual_laser_point = cv::Point2f((float)frame.cols * 0.5f, (float)frame.rows * 0.5f);
            play_session_time_sec = 0.0f;
            init_cat_play_state(&play_state);
            fprintf(stderr, "No cat%s; holding center servo=(%.2f,%.2f)\n",
                    inference_running ? " (inference busy)" : "",
                    servo_state.pan_deg, servo_state.tilt_deg);
        }

        cv::Mat detection = cv::imread("result.png");
        if (!detection.empty()) cv::imshow("YOLOv5 Live Detection", detection);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
        usleep(30000);
        frame_index++;
    }

    pthread_mutex_lock(&inference_shared.mutex);
    inference_shared.stop = 1;
    pthread_cond_signal(&inference_shared.cond);
    pthread_mutex_unlock(&inference_shared.mutex);
    pthread_join(inference_thread, NULL);
    pthread_mutex_destroy(&inference_shared.mutex);
    pthread_cond_destroy(&inference_shared.cond);

    // User requested behavior: laser turns off when Ctrl-C is received.
    if (g_sigint_received) {
        mosfet_gpio_set(&laser_gpio, false);
    }

    awnn_destroy(context);
    awnn_uninit();
    mosfet_gpio_close(&pan_power_gpio);
    mosfet_gpio_close(&tilt_power_gpio);
    mosfet_gpio_close(&laser_gpio);
    servo_pwm_close(&pan_pwm);
    servo_pwm_close(&tilt_pwm);
    camera.release();
    cv::destroyAllWindows();
    return 0;
}
