#ifndef YOLOV5_TRACKING_UTILS_H_
#define YOLOV5_TRACKING_UTILS_H_

#include <opencv2/core.hpp>
#include "yolov5_post_process.h"

struct LaserDotObservation {
    int detected;
    cv::Point2f center;
    float area;
};

struct LaserTrackState {
    int stable_detect_frames;
    int hold_miss_frames;
    // Set after we have emitted at least one confirmed/stable observation.
    int has_valid_lock;
    cv::Point2f last_center;
    float last_area;
};

struct CatTrackFilterState {
    int initialized;
    Yolov5CatTrackInfo filtered;
    int hold_miss_frames;
    // Identity stickiness state: suppress rapid target swaps when multiple
    // cats are present and confidence fluctuates frame-to-frame.
    int identity_lock_frames;
};

float clampf(float value, float min_v, float max_v);
LaserDotObservation detect_laser_dot(const cv::Mat &frame_bgr);
LaserDotObservation stabilize_laser_observation(LaserTrackState *state, const LaserDotObservation &raw);
Yolov5CatTrackInfo filter_cat_track(CatTrackFilterState *state, const Yolov5CatTrackInfo *raw_track);

#endif
