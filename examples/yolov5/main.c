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
#include <stdint.h>

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
    int has_scene_info;
    Yolov5SceneDetections latest_scene;
    double latest_scene_time_sec;
    uint64_t publish_generation;
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

enum SupervisorState {
    SUPERVISOR_IDLE = 0,
    SUPERVISOR_DETECT = 1,
    SUPERVISOR_WAKE = 2,
    SUPERVISOR_PLAY = 3,
    SUPERVISOR_DISENGAGE_TIMEOUT = 4,
    SUPERVISOR_SLEEP = 5,
};

enum PlayLoopMode {
    PLAY_LOOP_NONE = 0,
    PLAY_LOOP_CHASE = 1,
    PLAY_LOOP_BAIT = 2,
};

struct BaitState {
    float orbit_phase;
    float orbit_radius_x;
    float orbit_radius_y;
    int direction;
};

struct PersonWaveState {
    int initialized;
    int has_tracked_person;
    cv::Point2f last_center;
    int last_direction;
    int direction_flips;
    double last_motion_time_sec;
    double wave_latched_until_sec;
    Yolov5TrackedBox tracked_person;
};

struct SupervisorContext {
    enum SupervisorState state;
    enum PlayLoopMode play_mode;
    double state_since_sec;
    double cat_last_seen_sec;
    double human_last_seen_sec;
    double room_empty_since_sec;
    double wake_signal_since_sec;
    double last_detection_frame_sec;
    int cat_present_frames;
    int cat_absent_frames;
    int human_present_frames;
    int human_absent_frames;
    int room_empty_confirmed;
    int cat_present_confirmed;
    int human_present_confirmed;
    int wave_detected;
    cv::Point2f last_cat_center;
    Yolov5CatTrackInfo last_cat_track;
};

static const float RANDOM_MIN_SPEED_DEG_PER_SEC = 8.0f;
static const float RANDOM_MAX_SPEED_DEG_PER_SEC = 30.0f;

static void reset_wave_state(struct PersonWaveState *wave_state) {
    if (wave_state == NULL) {
        return;
    }
    memset(wave_state, 0, sizeof(*wave_state));
}

static void init_supervisor_context(struct SupervisorContext *supervisor, double now_sec) {
    if (supervisor == NULL) {
        return;
    }
    memset(supervisor, 0, sizeof(*supervisor));
    supervisor->state = SUPERVISOR_IDLE;
    supervisor->state_since_sec = now_sec;
    supervisor->play_mode = PLAY_LOOP_NONE;
    supervisor->cat_last_seen_sec = -1000.0;
    supervisor->human_last_seen_sec = -1000.0;
}

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

static double monotonic_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static const char *supervisor_state_name(enum SupervisorState state) {
    switch (state) {
        case SUPERVISOR_IDLE: return "IDLE";
        case SUPERVISOR_DETECT: return "DETECT";
        case SUPERVISOR_WAKE: return "WAKE";
        case SUPERVISOR_PLAY: return "PLAY";
        case SUPERVISOR_DISENGAGE_TIMEOUT: return "DISENGAGE_TIMEOUT";
        case SUPERVISOR_SLEEP: return "SLEEP";
        default: return "UNKNOWN";
    }
}

static void set_supervisor_state(struct SupervisorContext *ctx,
                                 enum SupervisorState next_state,
                                 double now_sec) {
    if (ctx->state == next_state) {
        return;
    }
    fprintf(stderr, "Supervisor transition: %s -> %s\n",
            supervisor_state_name(ctx->state),
            supervisor_state_name(next_state));
    ctx->state = next_state;
    ctx->state_since_sec = now_sec;
    if (next_state != SUPERVISOR_PLAY) {
        ctx->play_mode = PLAY_LOOP_NONE;
    }
    if (next_state == SUPERVISOR_DETECT || next_state == SUPERVISOR_WAKE) {
        ctx->wake_signal_since_sec = now_sec;
    }
}

static void init_bait_state(struct BaitState *bait_state) {
    bait_state->orbit_phase = 0.0f;
    bait_state->orbit_radius_x = 48.0f;
    bait_state->orbit_radius_y = 24.0f;
    bait_state->direction = 1;
}

static cv::Point2f clamp_point_to_frame_local(const cv::Point2f &p, int frame_w, int frame_h) {
    return cv::Point2f(clampf(p.x, 0.0f, (float)frame_w - 1.0f),
                       clampf(p.y, 0.0f, (float)frame_h - 1.0f));
}

static cv::Point2f tracked_box_center(const Yolov5TrackedBox *box) {
    return cv::Point2f(box->x + box->width * 0.5f,
                       box->y + box->height * 0.5f);
}

static float tracked_box_iou(const Yolov5TrackedBox *a, const Yolov5TrackedBox *b) {
    const float left = fmaxf(a->x, b->x);
    const float top = fmaxf(a->y, b->y);
    const float right = fminf(a->x + a->width, b->x + b->width);
    const float bottom = fminf(a->y + a->height, b->y + b->height);
    const float intersection_w = fmaxf(0.0f, right - left);
    const float intersection_h = fmaxf(0.0f, bottom - top);
    const float intersection_area = intersection_w * intersection_h;
    const float area_a = fmaxf(0.0f, a->width) * fmaxf(0.0f, a->height);
    const float area_b = fmaxf(0.0f, b->width) * fmaxf(0.0f, b->height);
    const float union_area = area_a + area_b - intersection_area;
    if (union_area <= 1e-5f) {
        return 0.0f;
    }
    return intersection_area / union_area;
}

static int select_wave_person_index(const struct PersonWaveState *wave_state,
                                    const Yolov5SceneDetections *scene,
                                    int *matched_prior_track) {
    if (matched_prior_track != NULL) {
        *matched_prior_track = 0;
    }
    if (scene == NULL || scene->person_count <= 0) {
        return -1;
    }
    if (wave_state == NULL || !wave_state->has_tracked_person) {
        return 0;
    }

    const cv::Point2f previous_center = tracked_box_center(&wave_state->tracked_person);
    const float distance_gate =
        fmaxf(wave_state->tracked_person.width, wave_state->tracked_person.height) * 0.75f + 90.0f;

    int best_index = 0;
    float best_score = -1.0f;
    float best_iou = 0.0f;
    float best_distance = 1e9f;
    for (int i = 0; i < scene->person_count; ++i) {
        const Yolov5TrackedBox *candidate = &scene->people[i];
        const cv::Point2f center = tracked_box_center(candidate);
        const float dx = center.x - previous_center.x;
        const float dy = center.y - previous_center.y;
        const float distance = sqrtf(dx * dx + dy * dy);
        const float iou = tracked_box_iou(&wave_state->tracked_person, candidate);
        const float score =
            candidate->confidence * 1.5f +
            iou * 3.0f +
            clampf(distance_gate - distance, 0.0f, distance_gate) * 0.01f;
        if (score > best_score) {
            best_score = score;
            best_index = i;
            best_iou = iou;
            best_distance = distance;
        }
    }

    if (best_iou > 0.0f && (best_iou > 0.08f || best_distance <= distance_gate)) {
        if (matched_prior_track != NULL) {
            *matched_prior_track = 1;
        }
    }
    return best_index;
}

static cv::Point2f build_bait_target(struct BaitState *bait_state,
                                     const struct SupervisorContext *ctx,
                                     int frame_w,
                                     int frame_h,
                                     float dt_sec) {
    cv::Point2f anchor((float)frame_w * 0.5f, (float)frame_h * 0.5f);
    if (ctx->last_cat_track.has_cat) {
        anchor.x = ctx->last_cat_track.x + ctx->last_cat_track.width * 0.5f;
        anchor.y = ctx->last_cat_track.y + ctx->last_cat_track.height * 0.35f;
        bait_state->orbit_radius_x = clampf(ctx->last_cat_track.width * 0.7f, 30.0f, 120.0f);
        bait_state->orbit_radius_y = clampf(ctx->last_cat_track.height * 0.45f, 18.0f, 70.0f);
    }

    bait_state->orbit_phase += dt_sec * 2.8f * (float)bait_state->direction;
    if (bait_state->orbit_phase > 6.28318f || bait_state->orbit_phase < -6.28318f) {
        bait_state->orbit_phase = 0.0f;
        bait_state->direction = -bait_state->direction;
    }

    cv::Point2f lure(anchor.x + cosf(bait_state->orbit_phase) * bait_state->orbit_radius_x,
                     anchor.y + sinf(bait_state->orbit_phase * 1.7f) * bait_state->orbit_radius_y);
    return clamp_point_to_frame_local(lure, frame_w, frame_h);
}

static void update_wave_state(struct PersonWaveState *wave_state,
                              const Yolov5SceneDetections *scene,
                              double now_sec,
                              int frame_w) {
    if (scene == NULL || scene->person_count <= 0) {
        wave_state->initialized = 0;
        wave_state->has_tracked_person = 0;
        wave_state->last_direction = 0;
        wave_state->direction_flips = 0;
        return;
    }

    int matched_prior_track = 0;
    const int person_index = select_wave_person_index(wave_state, scene, &matched_prior_track);
    if (person_index < 0) {
        wave_state->initialized = 0;
        wave_state->has_tracked_person = 0;
        wave_state->last_direction = 0;
        wave_state->direction_flips = 0;
        return;
    }

    const Yolov5TrackedBox *person = &scene->people[person_index];
    const cv::Point2f center = tracked_box_center(person);
    const int need_rebaseline =
        !wave_state->initialized || !wave_state->has_tracked_person || !matched_prior_track;
    if (need_rebaseline) {
        wave_state->initialized = 1;
        wave_state->has_tracked_person = 1;
        wave_state->tracked_person = *person;
        wave_state->last_center = center;
        wave_state->last_direction = 0;
        wave_state->direction_flips = 0;
        wave_state->last_motion_time_sec = now_sec;
        return;
    }

    const float delta_x = center.x - wave_state->last_center.x;
    const float normalized_motion = fabsf(delta_x) / fmaxf((float)frame_w, 1.0f);
    int direction = 0;
    if (normalized_motion > 0.035f) {
        direction = (delta_x > 0.0f) ? 1 : -1;
    }

    if (direction != 0) {
        if (wave_state->last_direction != 0 &&
            direction != wave_state->last_direction &&
            (now_sec - wave_state->last_motion_time_sec) < 1.25) {
            wave_state->direction_flips++;
        } else if ((now_sec - wave_state->last_motion_time_sec) > 1.25) {
            wave_state->direction_flips = 0;
        }
        wave_state->last_direction = direction;
        wave_state->last_motion_time_sec = now_sec;
    }

    if (wave_state->direction_flips >= 2) {
        wave_state->wave_latched_until_sec = now_sec + 1.5;
        wave_state->direction_flips = 0;
    }

    wave_state->tracked_person = *person;
    wave_state->last_center = center;
}

static void update_supervisor_presence(struct SupervisorContext *ctx,
                                       const Yolov5SceneDetections *scene,
                                       const Yolov5CatTrackInfo *tracked_cat,
                                       const struct PersonWaveState *wave_state,
                                       double now_sec) {
    const int cat_present = (tracked_cat != NULL && tracked_cat->has_cat);
    const int human_present = (scene != NULL && scene->person_count > 0);

    if (cat_present) {
        ctx->cat_present_frames = (ctx->cat_present_frames < 8) ? ctx->cat_present_frames + 1 : 8;
        ctx->cat_absent_frames = 0;
        ctx->cat_last_seen_sec = now_sec;
        ctx->cat_present_confirmed = (ctx->cat_present_frames >= 2);
        ctx->last_cat_track = *tracked_cat;
        ctx->last_cat_center = cv::Point2f(tracked_cat->x + tracked_cat->width * 0.5f,
                                           tracked_cat->y + tracked_cat->height * 0.5f);
    } else {
        ctx->cat_present_frames = 0;
        ctx->cat_absent_frames = (ctx->cat_absent_frames < 32) ? ctx->cat_absent_frames + 1 : 32;
        if (ctx->cat_absent_frames >= 8) {
            ctx->cat_present_confirmed = 0;
        }
    }

    if (human_present) {
        ctx->human_present_frames = (ctx->human_present_frames < 8) ? ctx->human_present_frames + 1 : 8;
        ctx->human_absent_frames = 0;
        ctx->human_last_seen_sec = now_sec;
        ctx->human_present_confirmed = (ctx->human_present_frames >= 2);
    } else {
        ctx->human_present_frames = 0;
        ctx->human_absent_frames = (ctx->human_absent_frames < 32) ? ctx->human_absent_frames + 1 : 32;
        if (ctx->human_absent_frames >= 8) {
            ctx->human_present_confirmed = 0;
        }
    }

    ctx->wave_detected = (wave_state != NULL && wave_state->wave_latched_until_sec > now_sec);
    ctx->room_empty_confirmed = !ctx->cat_present_confirmed && !ctx->human_present_confirmed;
    if (!ctx->room_empty_confirmed) {
        ctx->room_empty_since_sec = 0.0;
    } else if (ctx->room_empty_since_sec <= 0.0) {
        ctx->room_empty_since_sec = now_sec;
    }
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

static void run_servo_test_sequence(struct ServoPwm *pan_pwm,
                                    struct ServoPwm *tilt_pwm) {
    fprintf(stderr, "Servo test mode: drawing 3 smooth circles...\n");

    const float pan_radius_deg = 25.0f;
    const float tilt_radius_deg = 15.0f;
    const int steps_per_circle = 72;

    for (int cycle = 0; cycle < 3; ++cycle) {
        fprintf(stderr, "Servo test: circle %d/3\n", cycle + 1);
        for (int step = 0; step < steps_per_circle; ++step) {
            float t = (2.0f * (float)M_PI * (float)step) / (float)steps_per_circle;
            float pan = pan_radius_deg * cosf(t);
            float tilt = tilt_radius_deg * sinf(t);
            servo_pwm_set_angle(pan_pwm, pan);
            servo_pwm_set_angle(tilt_pwm, tilt);
            usleep(25000);
        }

        // Pause briefly between circles so each pass is visually distinct.
        sleep(1);
    }

    servo_pwm_set_angle(pan_pwm, 0.0f);
    servo_pwm_set_angle(tilt_pwm, 0.0f);
    fprintf(stderr, "Servo test mode complete.\n");
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
        const uint64_t publish_generation = args->shared->publish_generation;
        args->shared->has_new_frame = 0;
        args->shared->inference_running = 1;
        pthread_mutex_unlock(&args->shared->mutex);

        Yolov5CatTrackInfo track_info = {0, 0, 0, 0, 0, 0};
        Yolov5SceneDetections scene_info;
        memset(&scene_info, 0, sizeof(scene_info));

        if (!frame.empty() && cv::imwrite(args->frame_file, frame)) {
            unsigned int file_size = 0;
            unsigned char *plant_data = yolov5_pre_process(args->frame_file, &file_size);
            if (plant_data != NULL) {
                void *input_buffers[] = {plant_data};
                awnn_set_input_buffers(args->context, input_buffers);
                awnn_run(args->context);
                float **results = awnn_get_output_buffers(args->context);
                yolov5_post_process(args->frame_file, results, &track_info, &scene_info);
                free(plant_data);
            }
        }

        pthread_mutex_lock(&args->shared->mutex);
        if (publish_generation == args->shared->publish_generation) {
            args->shared->latest_track = track_info;
            args->shared->has_cat_info = 1;
            args->shared->latest_scene = scene_info;
            args->shared->has_scene_info = 1;
            args->shared->latest_scene_time_sec = monotonic_time_sec();
        }
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

    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <nbg> [camera_device] [laser_brightness_percent] [--test]\n"
                "  nbg: path to YOLOv5 .nb model\n"
                "  camera_device: optional V4L2 node (default: /dev/video0)\n"
                "  laser_brightness_percent: optional integer 0..100 (default: 100)\n"
                "  --test: run a 3-cycle smooth circular servo test and exit\n",
                argv[0]);
        return -1;
    }

    const char *nbg = argv[1];
    const char *camera_device = "/dev/video0";
    unsigned int laser_brightness_percent = 100;
    bool servo_test_mode = false;

    int positional_seen = 0;
    for (int i = 2; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--test") == 0) {
            servo_test_mode = true;
            continue;
        }

        if (positional_seen == 0) {
            // Backward compatible: allow ./yolov5 <nbg> <brightness> or <camera>.
            if (parse_brightness_percent(arg, &laser_brightness_percent) == 0) {
                positional_seen = 2;
                continue;
            }
            camera_device = arg;
            positional_seen = 1;
            continue;
        }

        if (positional_seen == 1) {
            if (parse_brightness_percent(arg, &laser_brightness_percent) == 0) {
                positional_seen = 2;
                continue;
            }

            fprintf(stderr,
                    "Invalid laser_brightness_percent '%s'. Expected integer 0..100.\n",
                    arg);
            return -1;
        }

        fprintf(stderr, "Unexpected argument '%s'.\n", arg);
        return -1;
    }

    const unsigned int laser_pwm_cycle_ticks = 10;
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

    srand((unsigned int)time(NULL));

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
    if (mosfet_gpio_open(&pan_power_gpio, mosfet_gpiochip_path, pan_power_gpio_line, false) < 0 ||
        mosfet_gpio_open(&tilt_power_gpio, mosfet_gpiochip_path, tilt_power_gpio_line, false) < 0 ||
        mosfet_gpio_open(&laser_gpio, mosfet_gpiochip_path, laser_gpio_line, false) < 0) {
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
        mosfet_gpio_set(&laser_gpio, false) < 0) {
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

    // Bare-minimum startup: center servos while laser remains safely off until armed.
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

    if (servo_test_mode) {
        run_servo_test_sequence(&pan_pwm, &tilt_pwm);
        mosfet_gpio_set(&laser_gpio, false);
        awnn_destroy(context);
        awnn_uninit();
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        mosfet_gpio_close(&laser_gpio);
        servo_pwm_close(&pan_pwm);
        servo_pwm_close(&tilt_pwm);
        camera.release();
        return 0;
    }

    struct InferenceShared inference_shared;
    pthread_mutex_init(&inference_shared.mutex, NULL);
    pthread_cond_init(&inference_shared.cond, NULL);
    inference_shared.has_new_frame = 0;
    inference_shared.stop = 0;
    inference_shared.inference_running = 0;
    inference_shared.has_cat_info = 0;
    inference_shared.latest_track.has_cat = 0;
    inference_shared.has_scene_info = 0;
    memset(&inference_shared.latest_scene, 0, sizeof(inference_shared.latest_scene));
    inference_shared.latest_scene_time_sec = 0.0;
    inference_shared.publish_generation = 1;

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
    CatPlayState play_state;
    init_cat_play_state(&play_state);
    MultiCatTrackState multi_cat_track = {0};
    LaserTrackState laser_track = {0};
    struct RandomScanState random_scan_state = {0};
    retarget_random_scan(&random_scan_state);
    struct BaitState bait_state;
    init_bait_state(&bait_state);
    struct PersonWaveState wave_state;
    reset_wave_state(&wave_state);
    struct SupervisorContext supervisor;
    init_supervisor_context(&supervisor, monotonic_time_sec());

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
                pthread_mutex_lock(&inference_shared.mutex);
                inference_shared.latest_frame.release();
                inference_shared.has_new_frame = 0;
                inference_shared.has_cat_info = 0;
                inference_shared.has_scene_info = 0;
                inference_shared.latest_track.has_cat = 0;
                memset(&inference_shared.latest_scene, 0, sizeof(inference_shared.latest_scene));
                inference_shared.latest_scene_time_sec = 0.0;
                inference_shared.publish_generation++;
                pthread_mutex_unlock(&inference_shared.mutex);
                multi_cat_track = (MultiCatTrackState){0};
                laser_track = (LaserTrackState){0};
                init_cat_play_state(&play_state);
                init_bait_state(&bait_state);
                reset_wave_state(&wave_state);
                init_supervisor_context(&supervisor, monotonic_time_sec());
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
            // Keep laser OFF until the supervisor re-arms play after recovery.
            mosfet_gpio_set(&laser_gpio, false);
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

        const double now_sec = monotonic_time_sec();
        const double state_elapsed_sec = now_sec - supervisor.state_since_sec;
        int inference_tick_divider = 1;
        switch (supervisor.state) {
            case SUPERVISOR_IDLE:
            case SUPERVISOR_SLEEP:
                inference_tick_divider = 6;
                break;
            case SUPERVISOR_DETECT:
            case SUPERVISOR_DISENGAGE_TIMEOUT:
                inference_tick_divider = 3;
                break;
            case SUPERVISOR_WAKE:
            case SUPERVISOR_PLAY:
            default:
                inference_tick_divider = 1;
                break;
        }

        static unsigned long frame_tick = 0;
        frame_tick++;
        if ((frame_tick % (unsigned long)inference_tick_divider) == 0) {
            pthread_mutex_lock(&inference_shared.mutex);
            inference_shared.latest_frame = frame.clone();
            inference_shared.has_new_frame = 1;
            pthread_cond_signal(&inference_shared.cond);
            pthread_mutex_unlock(&inference_shared.mutex);
        }

        pthread_mutex_lock(&inference_shared.mutex);
        Yolov5SceneDetections latest_scene = inference_shared.latest_scene;
        int has_track_info = inference_shared.has_cat_info;
        int has_scene_info = inference_shared.has_scene_info;
        double latest_scene_time_sec = inference_shared.latest_scene_time_sec;
        int inference_running = inference_shared.inference_running;
        pthread_mutex_unlock(&inference_shared.mutex);

        const double scene_fresh_timeout_sec =
            (supervisor.state == SUPERVISOR_IDLE || supervisor.state == SUPERVISOR_SLEEP) ? 1.0 : 0.5;
        const int scene_is_fresh = has_scene_info &&
            ((now_sec - latest_scene_time_sec) <= scene_fresh_timeout_sec);
        Yolov5SceneDetections *scene_ptr = scene_is_fresh ? &latest_scene : NULL;
        update_wave_state(&wave_state, scene_ptr, now_sec, frame.cols);

        Yolov5CatTrackInfo smoothed = select_multi_cat_track(&multi_cat_track, scene_ptr);
        if (!has_track_info && !scene_is_fresh) {
            smoothed.has_cat = 0;
        }

        update_supervisor_presence(&supervisor, scene_ptr, &smoothed, &wave_state, now_sec);
        const int wake_event_active = supervisor.cat_present_confirmed || supervisor.wave_detected;

        if (supervisor.state == SUPERVISOR_IDLE && wake_event_active) {
            set_supervisor_state(&supervisor, SUPERVISOR_DETECT, now_sec);
        } else if (supervisor.state == SUPERVISOR_DETECT) {
            if (wake_event_active) {
                if ((now_sec - supervisor.wake_signal_since_sec) >= 0.25) {
                    set_supervisor_state(&supervisor, SUPERVISOR_WAKE, now_sec);
                }
            } else if (state_elapsed_sec > 1.5) {
                set_supervisor_state(&supervisor, SUPERVISOR_IDLE, now_sec);
            }
        } else if (supervisor.state == SUPERVISOR_WAKE) {
            if (supervisor.cat_present_confirmed) {
                init_cat_play_state(&play_state);
                supervisor.play_mode = PLAY_LOOP_CHASE;
                set_supervisor_state(&supervisor, SUPERVISOR_PLAY, now_sec);
            } else if ((now_sec - supervisor.cat_last_seen_sec) <= 60.0 &&
                       supervisor.human_present_confirmed) {
                init_bait_state(&bait_state);
                supervisor.play_mode = PLAY_LOOP_BAIT;
                set_supervisor_state(&supervisor, SUPERVISOR_PLAY, now_sec);
            } else if (state_elapsed_sec > 3.0) {
                set_supervisor_state(&supervisor, SUPERVISOR_DISENGAGE_TIMEOUT, now_sec);
            }
        } else if (supervisor.state == SUPERVISOR_PLAY) {
            if (supervisor.cat_present_confirmed) {
                if (supervisor.play_mode != PLAY_LOOP_CHASE) {
                    init_cat_play_state(&play_state);
                }
                supervisor.play_mode = PLAY_LOOP_CHASE;
            } else if ((now_sec - supervisor.cat_last_seen_sec) <= 60.0 &&
                       supervisor.human_present_confirmed) {
                if (supervisor.play_mode != PLAY_LOOP_BAIT) {
                    init_bait_state(&bait_state);
                }
                supervisor.play_mode = PLAY_LOOP_BAIT;
            } else if (!supervisor.cat_present_confirmed &&
                       supervisor.human_present_confirmed &&
                       (now_sec - supervisor.cat_last_seen_sec) > 60.0) {
                set_supervisor_state(&supervisor, SUPERVISOR_DISENGAGE_TIMEOUT, now_sec);
            } else if (!supervisor.cat_present_confirmed && !supervisor.human_present_confirmed &&
                       (now_sec - fmax(supervisor.cat_last_seen_sec, supervisor.human_last_seen_sec)) >= 30.0) {
                set_supervisor_state(&supervisor, SUPERVISOR_DISENGAGE_TIMEOUT, now_sec);
            }
        } else if (supervisor.state == SUPERVISOR_DISENGAGE_TIMEOUT) {
            if (wake_event_active) {
                set_supervisor_state(&supervisor, SUPERVISOR_WAKE, now_sec);
            } else if (state_elapsed_sec > 10.0) {
                set_supervisor_state(&supervisor, SUPERVISOR_SLEEP, now_sec);
            }
        } else if (supervisor.state == SUPERVISOR_SLEEP) {
            if (state_elapsed_sec > 0.1) {
                set_supervisor_state(&supervisor, SUPERVISOR_IDLE, now_sec);
            }
        }

        LaserDotObservation raw_laser = detect_laser_dot(frame);
        LaserDotObservation stable_laser = stabilize_laser_observation(&laser_track, raw_laser);
        cv::Point2f current_laser = stable_laser.detected
            ? stable_laser.center
            : cv::Point2f((float)frame.cols * 0.5f, (float)frame.rows * 0.5f);

        cv::Point2f target_point((float)frame.cols * 0.5f, (float)frame.rows * 0.5f);
        const char *algo_name = "idle";
        bool laser_enabled = false;

        const bool outputs_allowed = !supervisor.room_empty_confirmed && !deadman_active;

        if (outputs_allowed &&
            supervisor.state == SUPERVISOR_PLAY &&
            supervisor.play_mode == PLAY_LOOP_CHASE &&
            smoothed.has_cat) {
            target_point = build_cat_play_target(&play_state, smoothed, current_laser,
                                                 (int)frame_tick, frame.cols, frame.rows,
                                                 control_dt_sec, &algo_name);
            laser_enabled = true;
        } else if (outputs_allowed &&
                   supervisor.state == SUPERVISOR_PLAY &&
                   supervisor.play_mode == PLAY_LOOP_BAIT &&
                   (now_sec - supervisor.cat_last_seen_sec) <= 60.0) {
            target_point = build_bait_target(&bait_state, &supervisor, frame.cols, frame.rows, control_dt_sec);
            algo_name = "bait";
            laser_enabled = true;
        } else if (outputs_allowed &&
                   (supervisor.state == SUPERVISOR_WAKE || supervisor.state == SUPERVISOR_DETECT)) {
            update_random_scan_servo(&servo_state, &random_scan_state, control_dt_sec);
            algo_name = "wake_scan";
        } else {
            servo_state.pan_deg = 0.0f;
            servo_state.tilt_deg = 0.0f;
        }

        if (supervisor.state == SUPERVISOR_PLAY && laser_enabled) {
            update_servo_state(&servo_state, current_laser, target_point, frame.cols, frame.rows);
        }
        servo_pwm_set_angle(&pan_pwm, servo_state.pan_deg);
        servo_pwm_set_angle(&tilt_pwm, servo_state.tilt_deg);

        if (!outputs_allowed || supervisor.state == SUPERVISOR_IDLE ||
            supervisor.state == SUPERVISOR_SLEEP) {
            laser_enabled = false;
        }

        const int laser_on_this_tick = laser_enabled && (laser_pwm_on_ticks > 0) &&
            (laser_pwm_tick < laser_pwm_on_ticks);
        mosfet_gpio_set(&laser_gpio, laser_on_this_tick);
        laser_pwm_tick = (laser_pwm_tick + 1) % laser_pwm_cycle_ticks;

        fprintf(stderr,
                "state=%s mode=%s cats=%d humans=%d wave=%d target=(%.1f,%.1f) servo=(%.2f,%.2f)%s\n",
                supervisor_state_name(supervisor.state),
                algo_name,
                scene_ptr ? scene_ptr->cat_count : 0,
                scene_ptr ? scene_ptr->person_count : 0,
                supervisor.wave_detected,
                target_point.x, target_point.y,
                servo_state.pan_deg, servo_state.tilt_deg,
                inference_running ? " inference_busy" : "");

        cv::Mat detection = cv::imread("result.png");
        if (!detection.empty()) cv::imshow("YOLOv5 Live Detection", detection);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
        usleep(30000);
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
