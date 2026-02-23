#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#if __has_include(<periphery/pwm.h>)
#include <periphery/pwm.h>
#else
#include <pwm.h>
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

struct LaserDotObservation {
    int detected;
    cv::Point2f center;
    float area;
};

static float clampf(float value, float min_v, float max_v) {
    return (value < min_v) ? min_v : ((value > max_v) ? max_v : value);
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
    printf("%s nbg [camera_device] [pan_pwm_chip] [pan_pwm_channel] [tilt_pwm_chip] [tilt_pwm_channel]\n", argv[0]);
    if (argc < 2) {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }

    const char *nbg = argv[1];
    const char *camera_device = (argc >= 3) ? argv[2] : "/dev/video0";
    const char *frame_file = "live_frame.jpg";

    // Defaults for two PWM-capable header pins from Radxa Cubie A7Z pinmux docs.
    // Override from argv when your board maps PWM differently.
    const unsigned int pan_pwm_chip = (argc >= 4) ? (unsigned int)atoi(argv[3]) : 0;
    const unsigned int pan_pwm_channel = (argc >= 5) ? (unsigned int)atoi(argv[4]) : 3;
    const unsigned int tilt_pwm_chip = (argc >= 6) ? (unsigned int)atoi(argv[5]) : 0;
    const unsigned int tilt_pwm_channel = (argc >= 7) ? (unsigned int)atoi(argv[6]) : 5;

    const int input_channels = 3;
    const int input_height = 480;
    const int input_width = 640;

    cv::VideoCapture camera(camera_device, cv::CAP_V4L2);
    if (!camera.isOpened()) {
        fprintf(stderr, "Failed to open webcam device: %s\n", camera_device);
        return -1;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, input_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, input_height);

    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);
    if (context == NULL) {
        fprintf(stderr, "Failed to create NPU context with nbg: %s\n", nbg);
        camera.release();
        awnn_uninit();
        return -1;
    }

    printf("Running live detection from %s\n", camera_device);
    printf("Enforcing input shape CxHxW = %dx%dx%d\n", input_channels, input_height, input_width);
    printf("Annotated detections will be written to result.png\n");
    printf("Displaying live detections in OpenCV window (press q to quit)\n");
    printf("PWM servo output pan=pwmchip%u:%u tilt=pwmchip%u:%u\n",
           pan_pwm_chip, pan_pwm_channel, tilt_pwm_chip, tilt_pwm_channel);

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
        servo_pwm_close(&pan_pwm);
        servo_pwm_close(&tilt_pwm);
        awnn_destroy(context);
        awnn_uninit();
        camera.release();
        return -1;
    }

    int printed_resolution = 0;
    ServoState servo_state = {0.0f, 0.0f};
    cv::Point2f estimated_laser((float)input_width * 0.5f, (float)input_height * 0.5f);
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
        }

        cv::Mat frame = raw_frame;
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        if (frame.cols != input_width || frame.rows != input_height) {
            cv::resize(frame, frame, cv::Size(input_width, input_height));
        }

        if (frame.channels() != input_channels) {
            fprintf(stderr, "Unexpected channel count after normalization: %d\n", frame.channels());
            usleep(100000);
            continue;
        }

        if (!cv::imwrite(frame_file, frame)) {
            fprintf(stderr, "Failed to write frame image: %s\n", frame_file);
            usleep(100000);
            continue;
        }

        unsigned int file_size = 0;
        unsigned char *plant_data = yolov5_pre_process(frame_file, &file_size);
        if (plant_data == NULL) {
            fprintf(stderr, "Pre-process failed for frame: %s\n", frame_file);
            usleep(100000);
            continue;
        }

        void *input_buffers[] = {plant_data};
        awnn_set_input_buffers(context, input_buffers);
        awnn_run(context);

        float **results = awnn_get_output_buffers(context);
        Yolov5CatTrackInfo track_info;
        yolov5_post_process(frame_file, results, &track_info);

        LaserDotObservation laser_obs = detect_laser_dot(frame);
        if (laser_obs.detected) {
            estimated_laser = laser_obs.center;
        }

        if (track_info.has_cat) {
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
            fprintf(stderr, "No cat detection; holding servo pan=%.2f tilt=%.2f\n",
                    servo_state.pan_deg,
                    servo_state.tilt_deg);
        }

        cv::Mat detection = cv::imread("result.png");
        if (!detection.empty()) {
            cv::imshow("YOLOv5 Live Detection", detection);
        }

        free(plant_data);
        frame_index++;

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }

        usleep(30000);
    }

    awnn_destroy(context);
    awnn_uninit();
    servo_pwm_close(&pan_pwm);
    servo_pwm_close(&tilt_pwm);
    camera.release();
    cv::destroyAllWindows();

    return 0;
}
