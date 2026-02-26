#ifndef YOLOV5_PLAY_ALGORITHMS_H_
#define YOLOV5_PLAY_ALGORITHMS_H_

#include <opencv2/core.hpp>
#include "yolov5_post_process.h"

enum CatPlayAlgorithm {
    CAT_PLAY_OVAL = 0,
    CAT_PLAY_STARE_DART = 1,
    CAT_PLAY_ZIGZAG_RETREAT = 2,
};

enum StareDartPhase {
    STARE_DART_HOLD = 0,
    STARE_DART_DART = 1,
};

enum ZigZagPhase {
    ZIGZAG_APPROACH = 0,
    ZIGZAG_SHAKE = 1,
    ZIGZAG_RETREAT = 2,
    ZIGZAG_RETURN = 3,
};

struct CatPlayState {
    enum CatPlayAlgorithm algorithm;

    cv::Point2f last_cat_center;
    float cat_still_time_sec;

    // Engagement: larger means cat is actively reacting to laser motion.
    float engagement_score;
    float prev_cat_laser_dist;
    float session_time_sec;
    float calm_time_sec;

    enum StareDartPhase stare_dart_phase;
    float stare_dart_hold_time_sec;
    cv::Point2f stare_dart_hold_point;
    cv::Point2f stare_dart_dart_target;

    enum ZigZagPhase zigzag_phase;
    float zigzag_phase_time_sec;
    cv::Point2f zigzag_front_point;
    cv::Point2f zigzag_retreat_point;
};

void init_cat_play_state(struct CatPlayState *state);
cv::Point2f build_cat_play_target(
    struct CatPlayState *state,
    const Yolov5CatTrackInfo &cat,
    const cv::Point2f &laser,
    int frame_index,
    int frame_w,
    int frame_h,
    float dt_sec,
    const char **algo_name_out);

#endif
