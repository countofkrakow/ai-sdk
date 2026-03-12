#ifndef YOLOV5_PLAY_ALGORITHMS_INTERNAL_H_
#define YOLOV5_PLAY_ALGORITHMS_INTERNAL_H_

#include <opencv2/core.hpp>
#include <stdlib.h>
#include <math.h>

struct CatPlayTuningConfig {
    struct {
        float base_scale;
        float min_radius;
        float max_radius;
        float confidence_norm_offset;
        float confidence_norm_scale;
        float confidence_scale_base;
        float confidence_scale_gain;
        float near_dist_diag_scale;
        float far_dist_diag_scale;
        float distance_scale_base;
        float distance_scale_gain;
        float clamp_min;
        float clamp_max;
    } catch_radius;

    struct {
        float speed_norm_divisor;
        float novelty_baseline;
        float novelty_recovery_rate;
        float novelty_stale_decay_rate;
        float novelty_active_fatigue_rate;
        float novelty_min_score;
        float oval_margin_scale;
        float oval_rx_min;
        float oval_rx_max;
        float oval_ry_min;
        float oval_ry_max;
        float oval_scaled_min;
        float oval_scaled_max;
        float near_miss_trigger_base;
        float near_miss_low_engagement_threshold;
        float near_miss_radius_min;
        float near_miss_radius_max;
        float near_miss_pause_min_base;
        float near_miss_pause_min_gain;
        float near_miss_pause_max_base;
        float near_miss_pause_max_gain;
        float near_miss_segment_min_sec;
        float near_miss_segment_max_sec;
        float near_miss_orbit_speed;
        float near_miss_direction_flip_percent;
        float near_miss_pause_challenge_min_gain;
        float near_miss_pause_challenge_max_gain;
        float near_miss_pass_bias_scale;
        float near_miss_radius_challenge_min_gain;
        float near_miss_radius_challenge_max_gain;
        float oval_flip_chance_min;
        float oval_flip_chance_max;
        float oval_flip_cooldown_min_sec;
        float oval_flip_cooldown_max_sec;
        float near_miss_initial_radius_scale;
    } algorithms;

    struct {
        float hold_min_sec;
        float hold_max_sec;
        float long_hold_min_sec;
        float long_hold_max_sec;
        float low_speed_threshold;
        float low_speed_dart_prob;
    } stare_dart;

    struct {
        float front_y_offset;
        float front_reach_threshold_px;
        float retreat_reach_threshold_px;
        float retreat_clearance_px;
        float amp_x_scale;
        float amp_x_min;
        float amp_x_max;
        float amp_y_scale;
        float amp_y_min;
        float amp_y_max;
        float wave_speed_base;
        float wave_speed_gain;
        float wave_y_freq;
        float shake_duration_base;
        float shake_duration_speed_gain;
        float challenge_arc_base;
        float challenge_arc_gain;
        float challenge_speed_gain;
        float challenge_speed_min;
        float challenge_speed_max;
        float challenge_shake_duration_gain;
        float challenge_shake_duration_min;
        float challenge_shake_duration_max;
        float arc_scale_min;
        float arc_scale_max;
        float amp_x_scaled_min;
        float amp_x_scaled_max;
        float amp_y_scaled_min;
        float amp_y_scaled_max;
    } zigzag;

    struct {
        float reaction_norm_divisor;
        float heading_alignment_gain;
        float near_target_radius_scale;
        float dwell_max_sec;
        float dwell_ramp_sec;
        float dwell_gain;
        float dwell_decay_rate;
        float catch_bonus;
        float disengaged_max_sec;
        float reengage_min_disengaged_sec;
        float reengage_norm_sec;
        float reengage_bonus_gain;
        float catch_attempt_decay;
        float catch_attempt_gain;
        float score_ema_keep;
        float score_ema_gain;
        float velocity_ema_keep;
        float velocity_ema_gain;
        float challenge_ema_keep;
        float challenge_ema_gain;
        float near_target_challenge_weight;
        float catch_challenge_weight;
    } engagement;

    struct {
        float close_chase_speed_threshold;
        float close_chase_dist_scale;
        float close_chase_decay_rate;
        float close_chase_trigger_sec;
        float close_chase_max_sec;
        float pause_min_sec;
        float pause_max_sec;
        float cooldown_min_sec;
        float cooldown_max_sec;
        float cooldown_max_clamp;
    } hesitation;

    struct {
        float still_distance_px;
        float still_switch_sec;
        float session_calm_trigger_sec;
        float calm_pause_min_sec;
        float calm_pause_max_sec;
        float oval_phase_base_step;
        float oval_phase_speed_gain;
        float oval_challenge_speed_gain;
        float oval_challenge_speed_min;
        float oval_challenge_speed_max;
        float oval_challenge_arc_base;
        float oval_challenge_arc_gain;
        float oval_challenge_arc_min;
        float oval_challenge_arc_max;
        float near_target_challenge_min;
        float near_target_challenge_max;
        float random_gate_scale;
        float random_gate_min;
        float random_gate_max;
        float transition_base_percent;
        float rounding_bias;
        float transition_weight_epsilon;
    } behavior;

    struct {
        float tease_director_bias;
        float chase_director_bias;
        float pounce_director_bias;
        float recover_director_bias;
        float tease_to_oval_switch_prob;
        float chase_oval_to_zigzag_switch_prob;
        float pounce_to_oval_switch_prob;
        float chase_zigzag_speed_scale;
        float chase_zigzag_duration_scale;
        float pounce_oval_speed_scale;
        float recover_oval_speed_scale;
        float recover_zigzag_speed_scale;
        float recover_zigzag_duration_scale;
        float director_duration_min_sec;
        float director_duration_max_sec;
        float director_w_tease_base;
        float director_w_tease_low_engage_gain;
        float director_w_chase_base;
        float director_w_chase_speed_gain;
        float director_w_pounce_base;
        float director_w_pounce_catch_gain;
        float director_w_pounce_low_speed_gain;
        float director_w_recover_base;
        float director_w_recover_speed_minus_engage_gain;
    } director;

    struct {
        float speed_scale_min;
        float speed_scale_gain;
        float arc_scale_gain;
        float hold_scale_gain;
        float dart_scale_min;
        float dart_scale_gain;
        float transition_scale_base;
        float transition_scale_gain;
        float zigzag_duration_low_conf_gain;
    } confidence_fallback;
};

extern CatPlayTuningConfig g_tuning;
extern int g_tuning_initialized;
void ensure_tuning_initialized(void);

inline float clampf_local(float value, float min_v, float max_v) {
    return (value < min_v) ? min_v : ((value > max_v) ? max_v : value);
}

inline float random_float_range(float min_v, float max_v) {
    const float r = (float)rand() / (float)RAND_MAX;
    return min_v + (max_v - min_v) * r;
}

inline float point_distance(const cv::Point2f &a, const cv::Point2f &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

#endif
