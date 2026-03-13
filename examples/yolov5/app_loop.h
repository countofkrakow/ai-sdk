#ifndef YOLOV5_APP_LOOP_H_
#define YOLOV5_APP_LOOP_H_

#include <signal.h>

#include "app_runtime.h"

struct FrameInputs {
    cv::Mat frame_bgr;
    unsigned long frame_seq;
    double timestamp_sec;
};

struct PerceptionState {
    Yolov5CatTrackInfo raw_track;
    Yolov5CatDetections raw_detections;
    Yolov5CatTrackInfo active_track;
    Yolov5CatTrackInfo filtered_track;
    int has_inference;
    int inference_running;
    unsigned long inference_source_seq;
};

struct ControlDecision {
    cv::Point2f target_point;
    const char *algorithm_name;
    float intensity_scale;
    float engagement_score;
    enum PlayDirectorIntent director_intent;
    int has_target;
};

struct ActuationCommand {
    float pan_deg;
    float tilt_deg;
    int laser_on;
    unsigned int effective_on_ticks;
};

int app_run_loop(AppRuntime *rt, volatile sig_atomic_t *stop_flag);

#endif
