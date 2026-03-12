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

enum NearMissTeasePhase {
    NEAR_MISS_OFF = 0,
    NEAR_MISS_BURST = 1,
    NEAR_MISS_PAUSE = 2,
};

enum PlayDirectorIntent {
    DIRECTOR_INTENT_TEASE = 0,
    DIRECTOR_INTENT_CHASE = 1,
    DIRECTOR_INTENT_POUNCE_WINDOW = 2,
    DIRECTOR_INTENT_RECOVER = 3,
};

struct CatPlayState {
    enum CatPlayAlgorithm algorithm;

    cv::Point2f last_cat_center;
    float cat_still_time_sec;
    int velocity_initialized;
    cv::Point2f prev_velocity_cat_center;
    float cat_speed_px_per_sec_ema;
    int was_within_catch_radius;
    float recent_catch_attempt_score;

    // Engagement: larger means cat is actively reacting to laser motion.
    float engagement_score;
    // Per-algorithm engagement ranking used for transition probabilities.
    float algorithm_engagement_scores[3];
    float prev_cat_laser_dist;
    float session_time_sec;
    float calm_time_sec;
    float close_chase_time_sec;
    float hesitation_pause_time_sec;
    float hesitation_cooldown_sec;

    enum PlayDirectorIntent director_intent;
    float director_time_remaining_sec;

    // Oval-mode orbit state: direction is +1/-1 for cw/ccw style phase motion.
    float oval_phase;
    int oval_direction;
    float oval_direction_cooldown_sec;

    enum NearMissTeasePhase near_miss_phase;
    int near_miss_passes_remaining;
    float near_miss_angle_rad;
    float near_miss_radius_scale;
    float near_miss_segment_time_sec;
    float near_miss_segment_duration_sec;
    float near_miss_pause_time_sec;
    int near_miss_direction;
    cv::Point2f near_miss_pause_point;

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
