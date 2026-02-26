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
};

struct InferenceThreadArgs {
    Awnn_Context_t *context;
    const char *frame_file;
    struct InferenceShared *shared;
};

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

        if (!frame.empty() && cv::imwrite(args->frame_file, frame)) {
            unsigned int file_size = 0;
            unsigned char *plant_data = yolov5_pre_process(args->frame_file, &file_size);
            if (plant_data != NULL) {
                void *input_buffers[] = {plant_data};
                awnn_set_input_buffers(args->context, input_buffers);
                awnn_run(args->context);
                float **results = awnn_get_output_buffers(args->context);
                yolov5_post_process(args->frame_file, results, &track_info);
                free(plant_data);
            }
        }

        pthread_mutex_lock(&args->shared->mutex);
        args->shared->latest_track = track_info;
        args->shared->has_cat_info = 1;
        args->shared->inference_running = 0;
        pthread_mutex_unlock(&args->shared->mutex);
    }

    return NULL;
}

int main(int argc, char **argv) {
    printf("%s nbg [camera_device]\n", argv[0]);
    if (argc < 2) {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }

    const char *nbg = argv[1];
    const char *camera_device = (argc >= 3) ? argv[2] : "/dev/video0";
    const char *inference_frame_file = "live_frame.jpg";

    const unsigned int pan_pwm_chip = 1;
    const unsigned int pan_pwm_channel = 5;
    const unsigned int tilt_pwm_chip = 1;
    const unsigned int tilt_pwm_channel = 4;

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
    if (mosfet_gpio_open(&pan_power_gpio, mosfet_gpiochip_path, pan_power_gpio_line) < 0 ||
        mosfet_gpio_open(&tilt_power_gpio, mosfet_gpiochip_path, tilt_power_gpio_line) < 0 ||
        mosfet_gpio_open(&laser_gpio, mosfet_gpiochip_path, laser_gpio_line) < 0) {
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }
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

    run_laser_alignment_sequence(&pan_pwm, &tilt_pwm, &laser_gpio);

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
    struct RandomScanState random_scan = {0.0f, 0.0f, RANDOM_MIN_SPEED_DEG_PER_SEC, 0};
    retarget_random_scan(&random_scan);

    struct CatPlayState cat_play_state;
    init_cat_play_state(&cat_play_state);

    CatTrackFilterState track_filter = {0};
    LaserTrackState laser_track_state = {0};

    // Deadman: if camera stream stalls, center servos and cut power.
    time_t last_frame_time = time(NULL);

    int printed_resolution = 0;
    cv::Point2f estimated_laser(0.0f, 0.0f);
    int frame_index = 0;

    while (1) {
        cv::Mat raw_frame;
        if (!camera.read(raw_frame) || raw_frame.empty()) {
            if (difftime(time(NULL), last_frame_time) > 2.0) {
                // Safety: no fresh camera for >2s, stop motion and power rails.
                servo_pwm_set_angle(&pan_pwm, 0.0f);
                servo_pwm_set_angle(&tilt_pwm, 0.0f);
                mosfet_gpio_set(&pan_power_gpio, false);
                mosfet_gpio_set(&tilt_power_gpio, false);
                mosfet_gpio_set(&laser_gpio, false);
            }
            usleep(100000);
            continue;
        }

        if (!printed_resolution) {
            printed_resolution = 1;
            estimated_laser = cv::Point2f((float)raw_frame.cols * 0.5f, (float)raw_frame.rows * 0.5f);
        }

        cv::Mat frame = raw_frame;
        if (frame.channels() == 4) cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        else if (frame.channels() == 1) cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        if (frame.channels() != input_channels) continue;

        LaserDotObservation laser_raw = detect_laser_dot(frame);
        LaserDotObservation laser_obs = stabilize_laser_observation(&laser_track_state, laser_raw);
        if (laser_obs.detected) estimated_laser = laser_obs.center;

        last_frame_time = time(NULL);

        pthread_mutex_lock(&inference_shared.mutex);
        inference_shared.latest_frame = frame.clone();
        inference_shared.has_new_frame = 1;
        pthread_cond_signal(&inference_shared.cond);

        Yolov5CatTrackInfo raw_track = inference_shared.latest_track;
        int has_track_info = inference_shared.has_cat_info;
        int inference_running = inference_shared.inference_running;
        pthread_mutex_unlock(&inference_shared.mutex);

        Yolov5CatTrackInfo smoothed = filter_cat_track(&track_filter, (has_track_info ? &raw_track : NULL));

        if (smoothed.has_cat) {
            const char *algo_name = "oval";
            cv::Point2f play_target = build_cat_play_target(
                &cat_play_state,
                smoothed,
                estimated_laser,
                frame_index,
                frame.cols,
                frame.rows,
                control_dt_sec,
                &algo_name);
            update_servo_state(&servo_state, estimated_laser, play_target, frame.cols, frame.rows);
            servo_pwm_set_angle(&pan_pwm, servo_state.pan_deg);
            servo_pwm_set_angle(&tilt_pwm, servo_state.tilt_deg);

            // Keep laser off during calm-pause windows to reduce overstimulation.
            mosfet_gpio_set(&laser_gpio, (strcmp(algo_name, "calm_pause") != 0));

            fprintf(stderr,
                    "cat_conf=%.2f algo=%s laser=(%.1f,%.1f) target=(%.1f,%.1f) servo=(%.2f,%.2f)\n",
                    smoothed.confidence, algo_name,
                    estimated_laser.x, estimated_laser.y,
                    play_target.x, play_target.y,
                    servo_state.pan_deg, servo_state.tilt_deg);
        } else {
            update_random_scan_servo(&servo_state, &random_scan, control_dt_sec);
            servo_pwm_set_angle(&pan_pwm, servo_state.pan_deg);
            mosfet_gpio_set(&laser_gpio, true);
            servo_pwm_set_angle(&tilt_pwm, servo_state.tilt_deg);
            fprintf(stderr, "No cat%s; random scan servo=(%.2f,%.2f)\n",
                    inference_running ? " (inference busy)" : "",
                    servo_state.pan_deg, servo_state.tilt_deg);
        }

        cv::Mat detection = cv::imread("result.png");
        if (!detection.empty()) cv::imshow("YOLOv5 Live Detection", detection);

        frame_index++;
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
