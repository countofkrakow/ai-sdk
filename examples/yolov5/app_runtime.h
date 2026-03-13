#ifndef YOLOV5_APP_RUNTIME_H_
#define YOLOV5_APP_RUNTIME_H_

#include <pthread.h>
#include <vector>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <awnn_lib.h>

#include "app_config.h"
#include "debug_trace.h"
#include "play_engine.h"
#include "servo_control.h"
#include "tracking_utils.h"
#include "yolov5_post_process.h"

struct FrameMailbox {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    cv::Mat latest_frame;
    unsigned long frame_seq;
    double timestamp_sec;
    int has_new_frame;
    int stop;
    int inference_running;
};

struct InferenceMailbox {
    pthread_mutex_t mutex;
    Yolov5CatTrackInfo latest_track;
    Yolov5CatDetections latest_detections;
    unsigned long source_frame_seq;
    double source_timestamp_sec;
    int has_cat_info;
};

struct ReplayState {
    std::vector<cv::String> frame_paths;
    size_t index;
    int enabled;
};

struct AppRuntime {
    AppConfig cfg;
    cv::VideoCapture camera;
    Awnn_Context_t *context;

    MosfetPowerGpio pan_power_gpio;
    MosfetPowerGpio tilt_power_gpio;
    MosfetPowerGpio laser_gpio;
    ServoPwm pan_pwm;
    ServoPwm tilt_pwm;

    unsigned int laser_pwm_tick;
    unsigned int laser_pwm_on_ticks;

    FrameMailbox frame_mailbox;
    InferenceMailbox inference_mailbox;
    ReplayState replay;

    MultiCatTrackerState multi_cat_tracker;
    CatTrackFilterState track_filter;
    ServoState servo_state;
    cv::Point2f virtual_laser_point;
    PlayEngine *play_engine;
    float play_session_time_sec;
    int frame_index;

    int servo_rails_powered;
    int deadman_active;
    time_t last_frame_time;

    DebugTrace trace;
};

#endif
