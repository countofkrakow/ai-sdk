#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#if __has_include(<periphery/pwm.h>)
#include <periphery/pwm.h>
#include <periphery/gpio.h>
#else
#include <pwm.h>
#include <gpio.h>
#endif

#include <awnn_lib.h>

#include "image_utils.h"
#include "yolov5_pre_process.h"
#include "yolov5_post_process.h"

struct ServoState {
    float pan_deg;
    float tilt_deg;
};

struct ServoPwm {
    pwm_t *handle;
    unsigned int chip;
    unsigned int channel;
};

struct MosfetPowerGpio {
    gpio_t *handle;
    const char *chip_path;
    unsigned int line;
};

struct LaserDotObservation {
    int detected;
    cv::Point2f center;
    float area;
};

struct RandomScanState {
    float target_pan_deg;
    float target_tilt_deg;
    float speed_deg_per_sec;
    int frames_until_retarget;
};

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

// Keep these configurable limits for random "hunt" motion when no cat is detected.
static const float RANDOM_MIN_SPEED_DEG_PER_SEC = 8.0f;
static const float RANDOM_MAX_SPEED_DEG_PER_SEC = 30.0f;

static float clampf(float value, float min_v, float max_v) {
    return (value < min_v) ? min_v : ((value > max_v) ? max_v : value);
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

        Yolov5CatTrackInfo track_info;
        track_info.has_cat = 0;
        track_info.confidence = 0.0f;
        track_info.x = 0.0f;
        track_info.y = 0.0f;
        track_info.width = 0.0f;
        track_info.height = 0.0f;

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

static int servo_pwm_open(struct ServoPwm *servo_pwm, unsigned int chip, unsigned int channel) {
    servo_pwm->handle = pwm_new();
    if (servo_pwm->handle == NULL) {
        fprintf(stderr, "pwm_new failed for chip=%u channel=%u\n", chip, channel);
        return -1;
    }

    servo_pwm->chip = chip;
    servo_pwm->channel = channel;

    if (pwm_open(servo_pwm->handle, chip, channel) < 0) {
        fprintf(stderr, "pwm_open failed for chip=%u channel=%u\n", chip, channel);
        pwm_free(servo_pwm->handle);
        servo_pwm->handle = NULL;
        return -1;
    }

    // Typical hobby servo: 50Hz => 20ms period.
    if (pwm_set_period_ns(servo_pwm->handle, 20000000ULL) < 0) {
        fprintf(stderr, "pwm_set_period_ns failed for chip=%u channel=%u\n", chip, channel);
        pwm_close(servo_pwm->handle);
        pwm_free(servo_pwm->handle);
        servo_pwm->handle = NULL;
        return -1;
    }

    return 0;
}

static int servo_pwm_set_angle(struct ServoPwm *servo_pwm, float angle_deg) {
    if (servo_pwm->handle == NULL) {
        return -1;
    }

    // Map [-45,+45] to [1000us,2000us].
    const float clamped = clampf(angle_deg, -45.0f, 45.0f);
    const float ratio = (clamped + 45.0f) / 90.0f;
    const unsigned long long pulse_ns = (unsigned long long)(1000000.0f + ratio * 1000000.0f);

    if (pwm_set_duty_cycle_ns(servo_pwm->handle, pulse_ns) < 0) {
        fprintf(stderr, "pwm_set_duty_cycle_ns failed for chip=%u channel=%u\n", servo_pwm->chip, servo_pwm->channel);
        return -1;
    }

    return 0;
}

static int servo_pwm_enable(struct ServoPwm *servo_pwm) {
    if (servo_pwm->handle == NULL) {
        return -1;
    }

    if (pwm_enable(servo_pwm->handle) < 0) {
        fprintf(stderr, "pwm_enable failed for chip=%u channel=%u\n", servo_pwm->chip, servo_pwm->channel);
        return -1;
    }

    return 0;
}

static int mosfet_gpio_open(struct MosfetPowerGpio *mosfet_gpio, const char *chip_path, unsigned int line) {
    mosfet_gpio->handle = gpio_new();
    if (mosfet_gpio->handle == NULL) {
        fprintf(stderr, "gpio_new failed for %s line=%u\n", chip_path, line);
        return -1;
    }

    mosfet_gpio->chip_path = chip_path;
    mosfet_gpio->line = line;

    if (gpio_open(mosfet_gpio->handle, chip_path, line, GPIO_DIR_OUT_LOW) < 0) {
        fprintf(stderr, "gpio_open failed for %s line=%u\n", chip_path, line);
        gpio_free(mosfet_gpio->handle);
        mosfet_gpio->handle = NULL;
        return -1;
    }

    return 0;
}

static int mosfet_gpio_set(struct MosfetPowerGpio *mosfet_gpio, bool enabled) {
    if (mosfet_gpio->handle == NULL) {
        return -1;
    }

    if (gpio_write(mosfet_gpio->handle, enabled) < 0) {
        fprintf(stderr, "gpio_write failed for %s line=%u\n", mosfet_gpio->chip_path, mosfet_gpio->line);
        return -1;
    }

    return 0;
}

static void mosfet_gpio_close(struct MosfetPowerGpio *mosfet_gpio) {
    if (mosfet_gpio->handle == NULL) {
        return;
    }

    gpio_write(mosfet_gpio->handle, false);
    gpio_close(mosfet_gpio->handle);
    gpio_free(mosfet_gpio->handle);
    mosfet_gpio->handle = NULL;
}

static void servo_pwm_close(struct ServoPwm *servo_pwm) {
    if (servo_pwm->handle == NULL) {
        return;
    }

    pwm_disable(servo_pwm->handle);
    pwm_close(servo_pwm->handle);
    pwm_free(servo_pwm->handle);
    servo_pwm->handle = NULL;
}

static LaserDotObservation detect_laser_dot(const cv::Mat &frame_bgr) {
    LaserDotObservation observation = {0, cv::Point2f(0.0f, 0.0f), 0.0f};
    if (frame_bgr.empty()) {
        return observation;
    }

    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);

    // Two red hue bands in HSV; tuned for bright, saturated laser spots.
    cv::Mat mask_low, mask_high, red_mask;
    cv::inRange(hsv, cv::Scalar(0, 120, 170), cv::Scalar(12, 255, 255), mask_low);
    cv::inRange(hsv, cv::Scalar(170, 120, 170), cv::Scalar(179, 255, 255), mask_high);
    cv::bitwise_or(mask_low, mask_high, red_mask);

    cv::GaussianBlur(red_mask, red_mask, cv::Size(3, 3), 0.0);
    cv::threshold(red_mask, red_mask, 200, 255, cv::THRESH_BINARY);
    cv::morphologyEx(
        red_mask, red_mask, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float best_area = 0.0f;
    cv::Point2f best_center;
    for (size_t i = 0; i < contours.size(); ++i) {
        const float area = (float)cv::contourArea(contours[i]);
        if (area < 2.0f || area > 250.0f) {
            continue;
        }

        cv::Moments m = cv::moments(contours[i]);
        if (fabs(m.m00) < 1e-5f) {
            continue;
        }

        cv::Point2f c((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
        if (area > best_area) {
            best_area = area;
            best_center = c;
        }
    }

    if (best_area > 0.0f) {
        observation.detected = 1;
        observation.center = best_center;
        observation.area = best_area;
    }

    return observation;
}

static cv::Point2f build_circle_target(const Yolov5CatTrackInfo &cat, int frame_index) {
    const float center_x = cat.x + cat.width * 0.5f;
    const float center_y = cat.y + cat.height * 0.5f;
    const float radius = clampf(fminf(cat.width, cat.height) * 0.28f, 12.0f, 65.0f);

    // Slow circular motion around cat center.
    const float angular_step_rad = 0.18f;
    const float theta = frame_index * angular_step_rad;

    return cv::Point2f(
        center_x + radius * cosf(theta),
        center_y + radius * sinf(theta));
}

static void update_servo_state(
    ServoState *servo_state,
    const cv::Point2f &current_laser,
    const cv::Point2f &target,
    int frame_width,
    int frame_height) {
    // Small proportional controller from pixel error to servo angle delta.
    const float pan_gain = 16.0f;
    const float tilt_gain = 16.0f;
    const float max_step_deg = 1.8f;

    float err_x = (target.x - current_laser.x) / (float)frame_width;
    float err_y = (target.y - current_laser.y) / (float)frame_height;

    float pan_delta = clampf(err_x * pan_gain, -max_step_deg, max_step_deg);
    float tilt_delta = clampf(err_y * tilt_gain, -max_step_deg, max_step_deg);

    servo_state->pan_deg = clampf(servo_state->pan_deg + pan_delta, -45.0f, 45.0f);
    servo_state->tilt_deg = clampf(servo_state->tilt_deg + tilt_delta, -45.0f, 45.0f);
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

    // Fixed hardware mapping for Radxa Cubie A7Z from 40-pin header table:
    //   pan  -> pin 3  (PJ23 / PWM1-5) => pwmchip1 channel 5
    //   tilt -> pin 5  (PJ22 / PWM1-4) => pwmchip1 channel 4
    const unsigned int pan_pwm_chip = 1;
    const unsigned int pan_pwm_channel = 5;
    const unsigned int tilt_pwm_chip = 1;
    const unsigned int tilt_pwm_channel = 4;

    // MOSFET power control lines (A7Z header GPIOs):
    //   pan  power -> pin 7  (PB0)  => gpiochip0 line 32
    //   tilt power -> pin 11 (PB1)  => gpiochip0 line 33
    const char *mosfet_gpiochip_path = "/dev/gpiochip0";
    const unsigned int pan_power_gpio_line = 32;
    const unsigned int tilt_power_gpio_line = 33;

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

    printf("Running live detection from %s\n", camera_device);
    printf("Using native camera resolution; YOLO pre-process handles letterbox scaling\n");
    printf("Annotated detections will be written to result.png\n");
    printf("Displaying live detections in OpenCV window (press q to quit)\n");
    printf("PWM servo output pan=pwmchip%u:%u tilt=pwmchip%u:%u\n",
           pan_pwm_chip, pan_pwm_channel, tilt_pwm_chip, tilt_pwm_channel);
    printf("Servo MOSFET power control pan=%s:%u tilt=%s:%u\n",
           mosfet_gpiochip_path, pan_power_gpio_line, mosfet_gpiochip_path, tilt_power_gpio_line);

    struct MosfetPowerGpio pan_power_gpio = {0};
    struct MosfetPowerGpio tilt_power_gpio = {0};
    if (mosfet_gpio_open(&pan_power_gpio, mosfet_gpiochip_path, pan_power_gpio_line) < 0 ||
        mosfet_gpio_open(&tilt_power_gpio, mosfet_gpiochip_path, tilt_power_gpio_line) < 0) {
        fprintf(stderr, "Failed to open MOSFET power GPIO outputs. Check gpiochip path/line and pinmux.\n");
        if (pan_power_gpio.handle) mosfet_gpio_close(&pan_power_gpio);
        if (tilt_power_gpio.handle) mosfet_gpio_close(&tilt_power_gpio);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }

    if (mosfet_gpio_set(&pan_power_gpio, true) < 0 ||
        mosfet_gpio_set(&tilt_power_gpio, true) < 0) {
        fprintf(stderr, "Failed to enable MOSFET servo power rails\n");
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }

    struct ServoPwm pan_pwm = {0};
    struct ServoPwm tilt_pwm = {0};
    if (servo_pwm_open(&pan_pwm, pan_pwm_chip, pan_pwm_channel) < 0 ||
        servo_pwm_open(&tilt_pwm, tilt_pwm_chip, tilt_pwm_channel) < 0) {
        fprintf(stderr, "Failed to open PWM outputs. Check pinmux and pwm chip/channel mapping.\n");
        if (pan_pwm.handle) servo_pwm_close(&pan_pwm);
        if (tilt_pwm.handle) servo_pwm_close(&tilt_pwm);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }

    if (servo_pwm_set_angle(&pan_pwm, 0.0f) < 0 ||
        servo_pwm_set_angle(&tilt_pwm, 0.0f) < 0 ||
        servo_pwm_enable(&pan_pwm) < 0 ||
        servo_pwm_enable(&tilt_pwm) < 0) {
        fprintf(stderr, "Failed to initialize servo PWM state\n");
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        servo_pwm_close(&pan_pwm);
        servo_pwm_close(&tilt_pwm);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }

    struct InferenceShared inference_shared;
    pthread_mutex_init(&inference_shared.mutex, NULL);
    pthread_cond_init(&inference_shared.cond, NULL);
    inference_shared.has_new_frame = 0;
    inference_shared.stop = 0;
    inference_shared.inference_running = 0;
    inference_shared.has_cat_info = 0;
    inference_shared.latest_track.has_cat = 0;

    struct InferenceThreadArgs worker_args;
    worker_args.context = context;
    worker_args.frame_file = inference_frame_file;
    worker_args.shared = &inference_shared;

    pthread_t inference_thread;
    if (pthread_create(&inference_thread, NULL, inference_thread_main, &worker_args) != 0) {
        fprintf(stderr, "Failed to create inference thread\n");
        mosfet_gpio_close(&pan_power_gpio);
        mosfet_gpio_close(&tilt_power_gpio);
        servo_pwm_close(&pan_pwm);
        servo_pwm_close(&tilt_pwm);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        pthread_mutex_destroy(&inference_shared.mutex);
        pthread_cond_destroy(&inference_shared.cond);
        return -1;
    }

    int printed_resolution = 0;
    ServoState servo_state = {0.0f, 0.0f};
    struct RandomScanState random_scan = {0.0f, 0.0f, RANDOM_MIN_SPEED_DEG_PER_SEC, 0};
    retarget_random_scan(&random_scan);
    cv::Point2f estimated_laser(0.0f, 0.0f);
    int frame_index = 0;

    while (1) {
        cv::Mat raw_frame;
        if (!camera.read(raw_frame) || raw_frame.empty()) {
            fprintf(stderr, "Failed to read frame from webcam\n");
            usleep(100000);
            continue;
        }

        if (!printed_resolution) {
            printf("Webcam frame resolution: %dx%d\n", raw_frame.cols, raw_frame.rows);
            printed_resolution = 1;
            estimated_laser = cv::Point2f((float)raw_frame.cols * 0.5f, (float)raw_frame.rows * 0.5f);
        }

        cv::Mat frame = raw_frame;
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        if (frame.channels() != input_channels) {
            fprintf(stderr, "Unexpected channel count after normalization: %d\n", frame.channels());
            usleep(100000);
            continue;
        }

        LaserDotObservation laser_obs = detect_laser_dot(frame);
        if (laser_obs.detected) {
            estimated_laser = laser_obs.center;
        }

        pthread_mutex_lock(&inference_shared.mutex);
        inference_shared.latest_frame = frame.clone();
        inference_shared.has_new_frame = 1;
        pthread_cond_signal(&inference_shared.cond);

        Yolov5CatTrackInfo track_info = inference_shared.latest_track;
        int has_track_info = inference_shared.has_cat_info;
        int inference_running = inference_shared.inference_running;
        pthread_mutex_unlock(&inference_shared.mutex);

        if (has_track_info && track_info.has_cat) {
            cv::Point2f circle_target = build_circle_target(track_info, frame_index);
            update_servo_state(&servo_state, estimated_laser, circle_target, frame.cols, frame.rows);
            servo_pwm_set_angle(&pan_pwm, servo_state.pan_deg);
            servo_pwm_set_angle(&tilt_pwm, servo_state.tilt_deg);

            fprintf(stderr,
                    "cat_conf=%.2f laser=(%.1f,%.1f) circle_target=(%.1f,%.1f) servo_pan=%.2f servo_tilt=%.2f\n",
                    track_info.confidence,
                    estimated_laser.x,
                    estimated_laser.y,
                    circle_target.x,
                    circle_target.y,
                    servo_state.pan_deg,
                    servo_state.tilt_deg);
        } else {
            update_random_scan_servo(&servo_state, &random_scan, control_dt_sec);
            servo_pwm_set_angle(&pan_pwm, servo_state.pan_deg);
            servo_pwm_set_angle(&tilt_pwm, servo_state.tilt_deg);
            fprintf(stderr, "No cat detection%s; random scan pan=%.2f tilt=%.2f speed=%.2f target=(%.2f,%.2f)\n",
                    inference_running ? " (inference busy)" : "",
                    servo_state.pan_deg,
                    servo_state.tilt_deg,
                    random_scan.speed_deg_per_sec,
                    random_scan.target_pan_deg,
                    random_scan.target_tilt_deg);
        }

        cv::Mat detection = cv::imread("result.png");
        if (!detection.empty()) {
            cv::imshow("YOLOv5 Live Detection", detection);
        }

        frame_index++;

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }

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
    servo_pwm_close(&pan_pwm);
    servo_pwm_close(&tilt_pwm);
    camera.release();
    cv::destroyAllWindows();

    return 0;
}
