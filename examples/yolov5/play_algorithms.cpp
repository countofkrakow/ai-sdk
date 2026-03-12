#include "play_algorithms.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>

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

static CatPlayTuningConfig g_tuning;
static int g_tuning_initialized = 0;

void reset_cat_play_tuning_defaults(void) {
    g_tuning.catch_radius.base_scale = 0.18f;
    g_tuning.catch_radius.min_radius = 10.0f;
    g_tuning.catch_radius.max_radius = 42.0f;
    g_tuning.catch_radius.confidence_norm_offset = 0.25f;
    g_tuning.catch_radius.confidence_norm_scale = 0.60f;
    g_tuning.catch_radius.confidence_scale_base = 0.90f;
    g_tuning.catch_radius.confidence_scale_gain = 0.25f;
    g_tuning.catch_radius.near_dist_diag_scale = 1.1f;
    g_tuning.catch_radius.far_dist_diag_scale = 2.2f;
    g_tuning.catch_radius.distance_scale_base = 0.88f;
    g_tuning.catch_radius.distance_scale_gain = 0.20f;
    g_tuning.catch_radius.clamp_min = 8.0f;
    g_tuning.catch_radius.clamp_max = 52.0f;

    g_tuning.algorithms.speed_norm_divisor = 220.0f;
    g_tuning.algorithms.novelty_baseline = 0.50f;
    g_tuning.algorithms.novelty_recovery_rate = 0.04f;
    g_tuning.algorithms.novelty_stale_decay_rate = 0.012f;
    g_tuning.algorithms.novelty_active_fatigue_rate = 0.02f;
    g_tuning.algorithms.novelty_min_score = 0.05f;
    g_tuning.algorithms.oval_margin_scale = 1.15f;
    g_tuning.algorithms.oval_rx_min = 12.0f;
    g_tuning.algorithms.oval_rx_max = 140.0f;
    g_tuning.algorithms.oval_ry_min = 12.0f;
    g_tuning.algorithms.oval_ry_max = 140.0f;
    g_tuning.algorithms.oval_scaled_min = 10.0f;
    g_tuning.algorithms.oval_scaled_max = 220.0f;
    g_tuning.algorithms.near_miss_trigger_base = 0.08f;
    g_tuning.algorithms.near_miss_low_engagement_threshold = 0.45f;
    g_tuning.algorithms.near_miss_radius_min = 1.10f;
    g_tuning.algorithms.near_miss_radius_max = 1.40f;
    g_tuning.algorithms.near_miss_pause_min_base = 0.30f;
    g_tuning.algorithms.near_miss_pause_min_gain = 0.35f;
    g_tuning.algorithms.near_miss_pause_max_base = 0.65f;
    g_tuning.algorithms.near_miss_pause_max_gain = 0.55f;
    g_tuning.algorithms.near_miss_segment_min_sec = 0.16f;
    g_tuning.algorithms.near_miss_segment_max_sec = 0.34f;
    g_tuning.algorithms.near_miss_orbit_speed = 11.0f;
    g_tuning.algorithms.near_miss_direction_flip_percent = 65.0f;
    g_tuning.algorithms.near_miss_pause_challenge_min_gain = 0.22f;
    g_tuning.algorithms.near_miss_pause_challenge_max_gain = 0.26f;
    g_tuning.algorithms.near_miss_pass_bias_scale = 3.0f;
    g_tuning.algorithms.near_miss_radius_challenge_min_gain = 0.10f;
    g_tuning.algorithms.near_miss_radius_challenge_max_gain = 0.14f;
    g_tuning.algorithms.oval_flip_chance_min = 0.45f;
    g_tuning.algorithms.oval_flip_chance_max = 0.85f;
    g_tuning.algorithms.oval_flip_cooldown_min_sec = 0.4f;
    g_tuning.algorithms.oval_flip_cooldown_max_sec = 1.2f;
    g_tuning.algorithms.near_miss_initial_radius_scale = 1.2f;

    g_tuning.stare_dart.hold_min_sec = 2.0f;
    g_tuning.stare_dart.hold_max_sec = 6.0f;
    g_tuning.stare_dart.long_hold_min_sec = 5.0f;
    g_tuning.stare_dart.long_hold_max_sec = 30.0f;
    g_tuning.stare_dart.low_speed_threshold = 0.25f;
    g_tuning.stare_dart.low_speed_dart_prob = 0.02f;

    g_tuning.zigzag.front_y_offset = 0.25f;
    g_tuning.zigzag.front_reach_threshold_px = 18.0f;
    g_tuning.zigzag.retreat_reach_threshold_px = 20.0f;
    g_tuning.zigzag.retreat_clearance_px = 24.0f;
    g_tuning.zigzag.amp_x_scale = 0.45f;
    g_tuning.zigzag.amp_x_min = 12.0f;
    g_tuning.zigzag.amp_x_max = 80.0f;
    g_tuning.zigzag.amp_y_scale = 0.18f;
    g_tuning.zigzag.amp_y_min = 6.0f;
    g_tuning.zigzag.amp_y_max = 35.0f;
    g_tuning.zigzag.wave_speed_base = 6.0f;
    g_tuning.zigzag.wave_speed_gain = 6.0f;
    g_tuning.zigzag.wave_y_freq = 1.7f;
    g_tuning.zigzag.shake_duration_base = 3.2f;
    g_tuning.zigzag.shake_duration_speed_gain = 1.6f;
    g_tuning.zigzag.challenge_arc_base = 1.15f;
    g_tuning.zigzag.challenge_arc_gain = 0.30f;
    g_tuning.zigzag.challenge_speed_gain = 0.32f;
    g_tuning.zigzag.challenge_speed_min = 0.68f;
    g_tuning.zigzag.challenge_speed_max = 1.40f;
    g_tuning.zigzag.challenge_shake_duration_gain = 0.20f;
    g_tuning.zigzag.challenge_shake_duration_min = 0.6f;
    g_tuning.zigzag.challenge_shake_duration_max = 1.4f;
    g_tuning.zigzag.arc_scale_min = 0.72f;
    g_tuning.zigzag.arc_scale_max = 1.70f;
    g_tuning.zigzag.amp_x_scaled_min = 10.0f;
    g_tuning.zigzag.amp_x_scaled_max = 120.0f;
    g_tuning.zigzag.amp_y_scaled_min = 5.0f;
    g_tuning.zigzag.amp_y_scaled_max = 55.0f;

    g_tuning.engagement.reaction_norm_divisor = 25.0f;
    g_tuning.engagement.heading_alignment_gain = 0.20f;
    g_tuning.engagement.near_target_radius_scale = 1.4f;
    g_tuning.engagement.dwell_max_sec = 3.0f;
    g_tuning.engagement.dwell_ramp_sec = 1.2f;
    g_tuning.engagement.dwell_gain = 0.18f;
    g_tuning.engagement.dwell_decay_rate = 0.7f;
    g_tuning.engagement.catch_bonus = 0.35f;
    g_tuning.engagement.disengaged_max_sec = 10.0f;
    g_tuning.engagement.reengage_min_disengaged_sec = 0.25f;
    g_tuning.engagement.reengage_norm_sec = 3.5f;
    g_tuning.engagement.reengage_bonus_gain = 0.22f;
    g_tuning.engagement.catch_attempt_decay = 0.92f;
    g_tuning.engagement.catch_attempt_gain = 0.08f;
    g_tuning.engagement.score_ema_keep = 0.9f;
    g_tuning.engagement.score_ema_gain = 0.1f;
    g_tuning.engagement.velocity_ema_keep = 0.88f;
    g_tuning.engagement.velocity_ema_gain = 0.12f;
    g_tuning.engagement.challenge_ema_keep = 0.90f;
    g_tuning.engagement.challenge_ema_gain = 0.10f;
    g_tuning.engagement.near_target_challenge_weight = 0.16f;
    g_tuning.engagement.catch_challenge_weight = 0.20f;

    g_tuning.hesitation.close_chase_speed_threshold = 0.65f;
    g_tuning.hesitation.close_chase_dist_scale = 1.15f;
    g_tuning.hesitation.close_chase_decay_rate = 0.6f;
    g_tuning.hesitation.close_chase_trigger_sec = 0.35f;
    g_tuning.hesitation.close_chase_max_sec = 5.0f;
    g_tuning.hesitation.pause_min_sec = 0.2f;
    g_tuning.hesitation.pause_max_sec = 0.8f;
    g_tuning.hesitation.cooldown_min_sec = 1.4f;
    g_tuning.hesitation.cooldown_max_sec = 2.4f;
    g_tuning.hesitation.cooldown_max_clamp = 10.0f;

    g_tuning.behavior.still_distance_px = 10.0f;
    g_tuning.behavior.still_switch_sec = 60.0f;
    g_tuning.behavior.session_calm_trigger_sec = 120.0f;
    g_tuning.behavior.calm_pause_min_sec = 5.0f;
    g_tuning.behavior.calm_pause_max_sec = 12.0f;
    g_tuning.behavior.oval_phase_base_step = 0.10f;
    g_tuning.behavior.oval_phase_speed_gain = 0.16f;
    g_tuning.behavior.oval_challenge_speed_gain = 0.30f;
    g_tuning.behavior.oval_challenge_speed_min = 0.70f;
    g_tuning.behavior.oval_challenge_speed_max = 1.35f;
    g_tuning.behavior.oval_challenge_arc_base = 1.12f;
    g_tuning.behavior.oval_challenge_arc_gain = 0.28f;
    g_tuning.behavior.oval_challenge_arc_min = 0.75f;
    g_tuning.behavior.oval_challenge_arc_max = 1.55f;
    g_tuning.behavior.near_target_challenge_min = -1.0f;
    g_tuning.behavior.near_target_challenge_max = 1.0f;
    g_tuning.behavior.random_gate_scale = 30.0f;
    g_tuning.behavior.random_gate_min = 0.0f;
    g_tuning.behavior.random_gate_max = 2.0f;
    g_tuning.behavior.transition_base_percent = 10.0f;
    g_tuning.behavior.rounding_bias = 0.5f;
    g_tuning.behavior.transition_weight_epsilon = 0.0001f;

    g_tuning.director.tease_director_bias = 1.7f;
    g_tuning.director.chase_director_bias = 0.8f;
    g_tuning.director.pounce_director_bias = 0.7f;
    g_tuning.director.recover_director_bias = 0.55f;
    g_tuning.director.tease_to_oval_switch_prob = 0.15f;
    g_tuning.director.chase_oval_to_zigzag_switch_prob = 0.2f;
    g_tuning.director.pounce_to_oval_switch_prob = 0.25f;
    g_tuning.director.chase_zigzag_speed_scale = 1.30f;
    g_tuning.director.chase_zigzag_duration_scale = 0.75f;
    g_tuning.director.pounce_oval_speed_scale = 0.55f;
    g_tuning.director.recover_oval_speed_scale = 0.5f;
    g_tuning.director.recover_zigzag_speed_scale = 0.8f;
    g_tuning.director.recover_zigzag_duration_scale = 1.2f;
    g_tuning.director.director_duration_min_sec = 5.0f;
    g_tuning.director.director_duration_max_sec = 15.0f;
    g_tuning.director.director_w_tease_base = 0.22f;
    g_tuning.director.director_w_tease_low_engage_gain = 0.36f;
    g_tuning.director.director_w_chase_base = 0.18f;
    g_tuning.director.director_w_chase_speed_gain = 0.62f;
    g_tuning.director.director_w_pounce_base = 0.14f;
    g_tuning.director.director_w_pounce_catch_gain = 0.46f;
    g_tuning.director.director_w_pounce_low_speed_gain = 0.10f;
    g_tuning.director.director_w_recover_base = 0.16f;
    g_tuning.director.director_w_recover_speed_minus_engage_gain = 0.26f;

    g_tuning.confidence_fallback.speed_scale_min = 0.70f;
    g_tuning.confidence_fallback.speed_scale_gain = 0.55f;
    g_tuning.confidence_fallback.arc_scale_gain = 0.30f;
    g_tuning.confidence_fallback.hold_scale_gain = 1.10f;
    g_tuning.confidence_fallback.dart_scale_min = 0.35f;
    g_tuning.confidence_fallback.dart_scale_gain = 0.85f;
    g_tuning.confidence_fallback.transition_scale_base = 0.55f;
    g_tuning.confidence_fallback.transition_scale_gain = 0.90f;
    g_tuning.confidence_fallback.zigzag_duration_low_conf_gain = 0.45f;
    g_tuning_initialized = 1;
}

static void maybe_read_float(const cv::FileNode &n, float *dst) {
    if (!n.empty() && dst != NULL) {
        *dst = (float)n.real();
    }
}

int load_cat_play_tuning_json(const char *json_path) {
    reset_cat_play_tuning_defaults();
    if (json_path == NULL) {
        return -1;
    }

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) {
        return -1;
    }

    const cv::FileNode root = fs.root();
    const cv::FileNode cr = root["catch_radius"];
    maybe_read_float(cr["base_scale"], &g_tuning.catch_radius.base_scale);
    maybe_read_float(cr["min_radius"], &g_tuning.catch_radius.min_radius);
    maybe_read_float(cr["max_radius"], &g_tuning.catch_radius.max_radius);
    maybe_read_float(cr["confidence_norm_offset"], &g_tuning.catch_radius.confidence_norm_offset);
    maybe_read_float(cr["confidence_norm_scale"], &g_tuning.catch_radius.confidence_norm_scale);
    maybe_read_float(cr["confidence_scale_base"], &g_tuning.catch_radius.confidence_scale_base);
    maybe_read_float(cr["confidence_scale_gain"], &g_tuning.catch_radius.confidence_scale_gain);
    maybe_read_float(cr["near_dist_diag_scale"], &g_tuning.catch_radius.near_dist_diag_scale);
    maybe_read_float(cr["far_dist_diag_scale"], &g_tuning.catch_radius.far_dist_diag_scale);
    maybe_read_float(cr["distance_scale_base"], &g_tuning.catch_radius.distance_scale_base);
    maybe_read_float(cr["distance_scale_gain"], &g_tuning.catch_radius.distance_scale_gain);
    maybe_read_float(cr["clamp_min"], &g_tuning.catch_radius.clamp_min);
    maybe_read_float(cr["clamp_max"], &g_tuning.catch_radius.clamp_max);

    const cv::FileNode alg = root["algorithms"];
    const cv::FileNode common = alg["common"];
    maybe_read_float(common["speed_norm_divisor"], &g_tuning.algorithms.speed_norm_divisor);
    maybe_read_float(common["novelty_baseline"], &g_tuning.algorithms.novelty_baseline);
    maybe_read_float(common["novelty_recovery_rate"], &g_tuning.algorithms.novelty_recovery_rate);
    maybe_read_float(common["novelty_stale_decay_rate"], &g_tuning.algorithms.novelty_stale_decay_rate);
    maybe_read_float(common["novelty_active_fatigue_rate"], &g_tuning.algorithms.novelty_active_fatigue_rate);
    maybe_read_float(common["novelty_min_score"], &g_tuning.algorithms.novelty_min_score);

    const cv::FileNode oval = alg["oval"];
    maybe_read_float(oval["margin_scale"], &g_tuning.algorithms.oval_margin_scale);
    maybe_read_float(oval["rx_min"], &g_tuning.algorithms.oval_rx_min);
    maybe_read_float(oval["rx_max"], &g_tuning.algorithms.oval_rx_max);
    maybe_read_float(oval["ry_min"], &g_tuning.algorithms.oval_ry_min);
    maybe_read_float(oval["ry_max"], &g_tuning.algorithms.oval_ry_max);
    maybe_read_float(oval["scaled_min"], &g_tuning.algorithms.oval_scaled_min);
    maybe_read_float(oval["scaled_max"], &g_tuning.algorithms.oval_scaled_max);

    const cv::FileNode near_miss = alg["near_miss"];
    maybe_read_float(near_miss["trigger_base"], &g_tuning.algorithms.near_miss_trigger_base);
    maybe_read_float(near_miss["low_engagement_threshold"], &g_tuning.algorithms.near_miss_low_engagement_threshold);
    maybe_read_float(near_miss["radius_min"], &g_tuning.algorithms.near_miss_radius_min);
    maybe_read_float(near_miss["radius_max"], &g_tuning.algorithms.near_miss_radius_max);
    maybe_read_float(near_miss["pause_min_base"], &g_tuning.algorithms.near_miss_pause_min_base);
    maybe_read_float(near_miss["pause_min_gain"], &g_tuning.algorithms.near_miss_pause_min_gain);
    maybe_read_float(near_miss["pause_max_base"], &g_tuning.algorithms.near_miss_pause_max_base);
    maybe_read_float(near_miss["pause_max_gain"], &g_tuning.algorithms.near_miss_pause_max_gain);
    maybe_read_float(near_miss["segment_min_sec"], &g_tuning.algorithms.near_miss_segment_min_sec);
    maybe_read_float(near_miss["segment_max_sec"], &g_tuning.algorithms.near_miss_segment_max_sec);
    maybe_read_float(near_miss["orbit_speed"], &g_tuning.algorithms.near_miss_orbit_speed);
    maybe_read_float(near_miss["direction_flip_percent"], &g_tuning.algorithms.near_miss_direction_flip_percent);
    maybe_read_float(near_miss["pause_challenge_min_gain"], &g_tuning.algorithms.near_miss_pause_challenge_min_gain);
    maybe_read_float(near_miss["pause_challenge_max_gain"], &g_tuning.algorithms.near_miss_pause_challenge_max_gain);
    maybe_read_float(near_miss["pass_bias_scale"], &g_tuning.algorithms.near_miss_pass_bias_scale);
    maybe_read_float(near_miss["radius_challenge_min_gain"], &g_tuning.algorithms.near_miss_radius_challenge_min_gain);
    maybe_read_float(near_miss["radius_challenge_max_gain"], &g_tuning.algorithms.near_miss_radius_challenge_max_gain);
    maybe_read_float(near_miss["initial_radius_scale"], &g_tuning.algorithms.near_miss_initial_radius_scale);

    maybe_read_float(oval["flip_chance_min"], &g_tuning.algorithms.oval_flip_chance_min);
    maybe_read_float(oval["flip_chance_max"], &g_tuning.algorithms.oval_flip_chance_max);
    maybe_read_float(oval["flip_cooldown_min_sec"], &g_tuning.algorithms.oval_flip_cooldown_min_sec);
    maybe_read_float(oval["flip_cooldown_max_sec"], &g_tuning.algorithms.oval_flip_cooldown_max_sec);

    const cv::FileNode stare = alg["stare_dart"];
    maybe_read_float(stare["hold_min_sec"], &g_tuning.stare_dart.hold_min_sec);
    maybe_read_float(stare["hold_max_sec"], &g_tuning.stare_dart.hold_max_sec);
    maybe_read_float(stare["long_hold_min_sec"], &g_tuning.stare_dart.long_hold_min_sec);
    maybe_read_float(stare["long_hold_max_sec"], &g_tuning.stare_dart.long_hold_max_sec);
    maybe_read_float(stare["low_speed_threshold"], &g_tuning.stare_dart.low_speed_threshold);
    maybe_read_float(stare["low_speed_dart_prob"], &g_tuning.stare_dart.low_speed_dart_prob);

    const cv::FileNode zig = alg["zigzag"];
    maybe_read_float(zig["front_y_offset"], &g_tuning.zigzag.front_y_offset);
    maybe_read_float(zig["front_reach_threshold_px"], &g_tuning.zigzag.front_reach_threshold_px);
    maybe_read_float(zig["retreat_reach_threshold_px"], &g_tuning.zigzag.retreat_reach_threshold_px);
    maybe_read_float(zig["retreat_clearance_px"], &g_tuning.zigzag.retreat_clearance_px);
    maybe_read_float(zig["amp_x_scale"], &g_tuning.zigzag.amp_x_scale);
    maybe_read_float(zig["amp_x_min"], &g_tuning.zigzag.amp_x_min);
    maybe_read_float(zig["amp_x_max"], &g_tuning.zigzag.amp_x_max);
    maybe_read_float(zig["amp_y_scale"], &g_tuning.zigzag.amp_y_scale);
    maybe_read_float(zig["amp_y_min"], &g_tuning.zigzag.amp_y_min);
    maybe_read_float(zig["amp_y_max"], &g_tuning.zigzag.amp_y_max);
    maybe_read_float(zig["wave_speed_base"], &g_tuning.zigzag.wave_speed_base);
    maybe_read_float(zig["wave_speed_gain"], &g_tuning.zigzag.wave_speed_gain);
    maybe_read_float(zig["wave_y_freq"], &g_tuning.zigzag.wave_y_freq);
    maybe_read_float(zig["shake_duration_base"], &g_tuning.zigzag.shake_duration_base);
    maybe_read_float(zig["shake_duration_speed_gain"], &g_tuning.zigzag.shake_duration_speed_gain);
    maybe_read_float(zig["challenge_arc_base"], &g_tuning.zigzag.challenge_arc_base);
    maybe_read_float(zig["challenge_arc_gain"], &g_tuning.zigzag.challenge_arc_gain);
    maybe_read_float(zig["challenge_speed_gain"], &g_tuning.zigzag.challenge_speed_gain);
    maybe_read_float(zig["challenge_speed_min"], &g_tuning.zigzag.challenge_speed_min);
    maybe_read_float(zig["challenge_speed_max"], &g_tuning.zigzag.challenge_speed_max);
    maybe_read_float(zig["challenge_shake_duration_gain"], &g_tuning.zigzag.challenge_shake_duration_gain);
    maybe_read_float(zig["challenge_shake_duration_min"], &g_tuning.zigzag.challenge_shake_duration_min);
    maybe_read_float(zig["challenge_shake_duration_max"], &g_tuning.zigzag.challenge_shake_duration_max);
    maybe_read_float(zig["arc_scale_min"], &g_tuning.zigzag.arc_scale_min);
    maybe_read_float(zig["arc_scale_max"], &g_tuning.zigzag.arc_scale_max);
    maybe_read_float(zig["amp_x_scaled_min"], &g_tuning.zigzag.amp_x_scaled_min);
    maybe_read_float(zig["amp_x_scaled_max"], &g_tuning.zigzag.amp_x_scaled_max);
    maybe_read_float(zig["amp_y_scaled_min"], &g_tuning.zigzag.amp_y_scaled_min);
    maybe_read_float(zig["amp_y_scaled_max"], &g_tuning.zigzag.amp_y_scaled_max);

    const cv::FileNode engage = root["engagement"];
    maybe_read_float(engage["reaction_norm_divisor"], &g_tuning.engagement.reaction_norm_divisor);
    maybe_read_float(engage["heading_alignment_gain"], &g_tuning.engagement.heading_alignment_gain);
    maybe_read_float(engage["near_target_radius_scale"], &g_tuning.engagement.near_target_radius_scale);
    maybe_read_float(engage["dwell_max_sec"], &g_tuning.engagement.dwell_max_sec);
    maybe_read_float(engage["dwell_ramp_sec"], &g_tuning.engagement.dwell_ramp_sec);
    maybe_read_float(engage["dwell_gain"], &g_tuning.engagement.dwell_gain);
    maybe_read_float(engage["dwell_decay_rate"], &g_tuning.engagement.dwell_decay_rate);
    maybe_read_float(engage["catch_bonus"], &g_tuning.engagement.catch_bonus);
    maybe_read_float(engage["disengaged_max_sec"], &g_tuning.engagement.disengaged_max_sec);
    maybe_read_float(engage["reengage_min_disengaged_sec"], &g_tuning.engagement.reengage_min_disengaged_sec);
    maybe_read_float(engage["reengage_norm_sec"], &g_tuning.engagement.reengage_norm_sec);
    maybe_read_float(engage["reengage_bonus_gain"], &g_tuning.engagement.reengage_bonus_gain);
    maybe_read_float(engage["catch_attempt_decay"], &g_tuning.engagement.catch_attempt_decay);
    maybe_read_float(engage["catch_attempt_gain"], &g_tuning.engagement.catch_attempt_gain);
    maybe_read_float(engage["score_ema_keep"], &g_tuning.engagement.score_ema_keep);
    maybe_read_float(engage["score_ema_gain"], &g_tuning.engagement.score_ema_gain);
    maybe_read_float(engage["velocity_ema_keep"], &g_tuning.engagement.velocity_ema_keep);
    maybe_read_float(engage["velocity_ema_gain"], &g_tuning.engagement.velocity_ema_gain);
    maybe_read_float(engage["challenge_ema_keep"], &g_tuning.engagement.challenge_ema_keep);
    maybe_read_float(engage["challenge_ema_gain"], &g_tuning.engagement.challenge_ema_gain);
    maybe_read_float(engage["near_target_challenge_weight"], &g_tuning.engagement.near_target_challenge_weight);
    maybe_read_float(engage["catch_challenge_weight"], &g_tuning.engagement.catch_challenge_weight);

    const cv::FileNode hes = root["hesitation"];
    maybe_read_float(hes["close_chase_speed_threshold"], &g_tuning.hesitation.close_chase_speed_threshold);
    maybe_read_float(hes["close_chase_dist_scale"], &g_tuning.hesitation.close_chase_dist_scale);
    maybe_read_float(hes["close_chase_decay_rate"], &g_tuning.hesitation.close_chase_decay_rate);
    maybe_read_float(hes["close_chase_trigger_sec"], &g_tuning.hesitation.close_chase_trigger_sec);
    maybe_read_float(hes["close_chase_max_sec"], &g_tuning.hesitation.close_chase_max_sec);
    maybe_read_float(hes["pause_min_sec"], &g_tuning.hesitation.pause_min_sec);
    maybe_read_float(hes["pause_max_sec"], &g_tuning.hesitation.pause_max_sec);
    maybe_read_float(hes["cooldown_min_sec"], &g_tuning.hesitation.cooldown_min_sec);
    maybe_read_float(hes["cooldown_max_sec"], &g_tuning.hesitation.cooldown_max_sec);
    maybe_read_float(hes["cooldown_max_clamp"], &g_tuning.hesitation.cooldown_max_clamp);

    const cv::FileNode beh = root["behavior"];
    maybe_read_float(beh["still_distance_px"], &g_tuning.behavior.still_distance_px);
    maybe_read_float(beh["still_switch_sec"], &g_tuning.behavior.still_switch_sec);
    maybe_read_float(beh["session_calm_trigger_sec"], &g_tuning.behavior.session_calm_trigger_sec);
    maybe_read_float(beh["calm_pause_min_sec"], &g_tuning.behavior.calm_pause_min_sec);
    maybe_read_float(beh["calm_pause_max_sec"], &g_tuning.behavior.calm_pause_max_sec);
    maybe_read_float(beh["oval_phase_base_step"], &g_tuning.behavior.oval_phase_base_step);
    maybe_read_float(beh["oval_phase_speed_gain"], &g_tuning.behavior.oval_phase_speed_gain);
    maybe_read_float(beh["oval_challenge_speed_gain"], &g_tuning.behavior.oval_challenge_speed_gain);
    maybe_read_float(beh["oval_challenge_speed_min"], &g_tuning.behavior.oval_challenge_speed_min);
    maybe_read_float(beh["oval_challenge_speed_max"], &g_tuning.behavior.oval_challenge_speed_max);
    maybe_read_float(beh["oval_challenge_arc_base"], &g_tuning.behavior.oval_challenge_arc_base);
    maybe_read_float(beh["oval_challenge_arc_gain"], &g_tuning.behavior.oval_challenge_arc_gain);
    maybe_read_float(beh["oval_challenge_arc_min"], &g_tuning.behavior.oval_challenge_arc_min);
    maybe_read_float(beh["oval_challenge_arc_max"], &g_tuning.behavior.oval_challenge_arc_max);
    maybe_read_float(beh["near_target_challenge_min"], &g_tuning.behavior.near_target_challenge_min);
    maybe_read_float(beh["near_target_challenge_max"], &g_tuning.behavior.near_target_challenge_max);
    maybe_read_float(beh["random_gate_scale"], &g_tuning.behavior.random_gate_scale);
    maybe_read_float(beh["random_gate_min"], &g_tuning.behavior.random_gate_min);
    maybe_read_float(beh["random_gate_max"], &g_tuning.behavior.random_gate_max);
    maybe_read_float(beh["transition_base_percent"], &g_tuning.behavior.transition_base_percent);
    maybe_read_float(beh["rounding_bias"], &g_tuning.behavior.rounding_bias);
    maybe_read_float(beh["transition_weight_epsilon"], &g_tuning.behavior.transition_weight_epsilon);

    const cv::FileNode dir = root["director"];
    maybe_read_float(dir["tease_director_bias"], &g_tuning.director.tease_director_bias);
    maybe_read_float(dir["chase_director_bias"], &g_tuning.director.chase_director_bias);
    maybe_read_float(dir["pounce_director_bias"], &g_tuning.director.pounce_director_bias);
    maybe_read_float(dir["recover_director_bias"], &g_tuning.director.recover_director_bias);
    maybe_read_float(dir["tease_to_oval_switch_prob"], &g_tuning.director.tease_to_oval_switch_prob);
    maybe_read_float(dir["chase_oval_to_zigzag_switch_prob"], &g_tuning.director.chase_oval_to_zigzag_switch_prob);
    maybe_read_float(dir["pounce_to_oval_switch_prob"], &g_tuning.director.pounce_to_oval_switch_prob);
    maybe_read_float(dir["chase_zigzag_speed_scale"], &g_tuning.director.chase_zigzag_speed_scale);
    maybe_read_float(dir["chase_zigzag_duration_scale"], &g_tuning.director.chase_zigzag_duration_scale);
    maybe_read_float(dir["pounce_oval_speed_scale"], &g_tuning.director.pounce_oval_speed_scale);
    maybe_read_float(dir["recover_oval_speed_scale"], &g_tuning.director.recover_oval_speed_scale);
    maybe_read_float(dir["recover_zigzag_speed_scale"], &g_tuning.director.recover_zigzag_speed_scale);
    maybe_read_float(dir["recover_zigzag_duration_scale"], &g_tuning.director.recover_zigzag_duration_scale);
    maybe_read_float(dir["director_duration_min_sec"], &g_tuning.director.director_duration_min_sec);
    maybe_read_float(dir["director_duration_max_sec"], &g_tuning.director.director_duration_max_sec);
    maybe_read_float(dir["director_w_tease_base"], &g_tuning.director.director_w_tease_base);
    maybe_read_float(dir["director_w_tease_low_engage_gain"], &g_tuning.director.director_w_tease_low_engage_gain);
    maybe_read_float(dir["director_w_chase_base"], &g_tuning.director.director_w_chase_base);
    maybe_read_float(dir["director_w_chase_speed_gain"], &g_tuning.director.director_w_chase_speed_gain);
    maybe_read_float(dir["director_w_pounce_base"], &g_tuning.director.director_w_pounce_base);
    maybe_read_float(dir["director_w_pounce_catch_gain"], &g_tuning.director.director_w_pounce_catch_gain);
    maybe_read_float(dir["director_w_pounce_low_speed_gain"], &g_tuning.director.director_w_pounce_low_speed_gain);
    maybe_read_float(dir["director_w_recover_base"], &g_tuning.director.director_w_recover_base);
    maybe_read_float(dir["director_w_recover_speed_minus_engage_gain"], &g_tuning.director.director_w_recover_speed_minus_engage_gain);

    const cv::FileNode cf = root["confidence_fallback"];
    maybe_read_float(cf["speed_scale_min"], &g_tuning.confidence_fallback.speed_scale_min);
    maybe_read_float(cf["speed_scale_gain"], &g_tuning.confidence_fallback.speed_scale_gain);
    maybe_read_float(cf["arc_scale_gain"], &g_tuning.confidence_fallback.arc_scale_gain);
    maybe_read_float(cf["hold_scale_gain"], &g_tuning.confidence_fallback.hold_scale_gain);
    maybe_read_float(cf["dart_scale_min"], &g_tuning.confidence_fallback.dart_scale_min);
    maybe_read_float(cf["dart_scale_gain"], &g_tuning.confidence_fallback.dart_scale_gain);
    maybe_read_float(cf["transition_scale_base"], &g_tuning.confidence_fallback.transition_scale_base);
    maybe_read_float(cf["transition_scale_gain"], &g_tuning.confidence_fallback.transition_scale_gain);
    maybe_read_float(cf["zigzag_duration_low_conf_gain"], &g_tuning.confidence_fallback.zigzag_duration_low_conf_gain);
    g_tuning_initialized = 1;
    return 0;
}

static void ensure_tuning_initialized(void) {
    if (!g_tuning_initialized) {
        reset_cat_play_tuning_defaults();
    }
}

static float clampf_local(float value, float min_v, float max_v) {
    return (value < min_v) ? min_v : ((value > max_v) ? max_v : value);
}

static float random_float_range(float min_v, float max_v) {
    const float r = (float)rand() / (float)RAND_MAX;
    return min_v + (max_v - min_v) * r;
}

static float point_distance(const cv::Point2f &a, const cv::Point2f &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

static float compute_base_catch_radius(const Yolov5CatTrackInfo &cat) {
    return clampf_local(
        g_tuning.catch_radius.base_scale * (cat.width + cat.height),
        g_tuning.catch_radius.min_radius,
        g_tuning.catch_radius.max_radius);
}

static float compute_catch_radius(const Yolov5CatTrackInfo &cat, float cat_laser_dist, float behavior_confidence) {
    const float base = compute_base_catch_radius(cat);
    const float bbox_diag = sqrtf(cat.width * cat.width + cat.height * cat.height);

    // Lower confidence and larger cat-dot distance narrow the effective catch zone,
    // while high confidence and close-range play can widen it for challenge.
    const float confidence_norm = clampf_local(
        (behavior_confidence - g_tuning.catch_radius.confidence_norm_offset) /
            g_tuning.catch_radius.confidence_norm_scale,
        0.0f,
        1.0f);
    const float confidence_scale =
        g_tuning.catch_radius.confidence_scale_base + g_tuning.catch_radius.confidence_scale_gain * confidence_norm;

    const float near_dist = g_tuning.catch_radius.near_dist_diag_scale * bbox_diag;
    const float far_dist = g_tuning.catch_radius.far_dist_diag_scale * bbox_diag + 1.0f;
    const float close_norm = 1.0f - clampf_local((cat_laser_dist - near_dist) / (far_dist - near_dist), 0.0f, 1.0f);
    const float distance_scale =
        g_tuning.catch_radius.distance_scale_base + g_tuning.catch_radius.distance_scale_gain * close_norm;

    return clampf_local(
        base * confidence_scale * distance_scale,
        g_tuning.catch_radius.clamp_min,
        g_tuning.catch_radius.clamp_max);
}

static cv::Point2f clamp_point_to_frame(const cv::Point2f &p, int frame_w, int frame_h) {
    return cv::Point2f(
        clampf_local(p.x, 0.0f, (float)frame_w - 1.0f),
        clampf_local(p.y, 0.0f, (float)frame_h - 1.0f));
}

static int point_in_cat_bbox(const cv::Point2f &p, const Yolov5CatTrackInfo &cat) {
    return (p.x >= cat.x && p.x <= (cat.x + cat.width) &&
            p.y >= cat.y && p.y <= (cat.y + cat.height));
}

static cv::Point2f random_point_outside_cat(const Yolov5CatTrackInfo &cat, int frame_w, int frame_h) {
    for (int i = 0; i < 24; ++i) {
        cv::Point2f p(random_float_range(0.0f, (float)frame_w - 1.0f),
                      random_float_range(0.0f, (float)frame_h - 1.0f));
        if (!point_in_cat_bbox(p, cat)) {
            return p;
        }
    }

    return cv::Point2f(0.0f, 0.0f);
}

static cv::Point2f furthest_frame_corner_from_point(const cv::Point2f &p, int frame_w, int frame_h) {
    cv::Point2f corners[4] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f((float)frame_w - 1.0f, 0.0f),
        cv::Point2f(0.0f, (float)frame_h - 1.0f),
        cv::Point2f((float)frame_w - 1.0f, (float)frame_h - 1.0f),
    };

    int best = 0;
    float best_dist = -1.0f;
    for (int i = 0; i < 4; ++i) {
        float d = point_distance(corners[i], p);
        if (d > best_dist) {
            best = i;
            best_dist = d;
        }
    }

    return corners[best];
}

static int segment_intersects_cat_bbox(const cv::Point2f &a, const cv::Point2f &b, const Yolov5CatTrackInfo &cat) {
    cv::Rect2f rect(cat.x, cat.y, cat.width, cat.height);
    if (rect.contains(a) || rect.contains(b)) {
        return 1;
    }

    cv::Point p1((int)a.x, (int)a.y);
    cv::Point p2((int)b.x, (int)b.y);
    cv::Rect ir((int)cat.x, (int)cat.y, (int)cat.width, (int)cat.height);
    return cv::clipLine(ir, p1, p2);
}

static cv::Point2f build_retreat_target_avoiding_cat(
    const cv::Point2f &laser,
    const Yolov5CatTrackInfo &cat,
    int frame_w,
    int frame_h) {
    cv::Point2f furthest = furthest_frame_corner_from_point(laser, frame_w, frame_h);
    if (!segment_intersects_cat_bbox(laser, furthest, cat)) {
        return furthest;
    }

    const float clearance = g_tuning.zigzag.retreat_clearance_px;
    cv::Point2f waypoints[4] = {
        cv::Point2f(cat.x - clearance, cat.y - clearance),
        cv::Point2f(cat.x + cat.width + clearance, cat.y - clearance),
        cv::Point2f(cat.x - clearance, cat.y + cat.height + clearance),
        cv::Point2f(cat.x + cat.width + clearance, cat.y + cat.height + clearance),
    };

    int best = 0;
    float best_cost = FLT_MAX;
    for (int i = 0; i < 4; ++i) {
        cv::Point2f w = clamp_point_to_frame(waypoints[i], frame_w, frame_h);
        if (point_in_cat_bbox(w, cat)) {
            continue;
        }

        float cost = point_distance(laser, w) + point_distance(w, furthest);
        if (cost < best_cost) {
            best_cost = cost;
            best = i;
        }
    }

    return clamp_point_to_frame(waypoints[best], frame_w, frame_h);
}

static enum CatPlayAlgorithm pick_other_algorithm(enum CatPlayAlgorithm current) {
    int r = rand() % 2;
    if (current == CAT_PLAY_OVAL) return r == 0 ? CAT_PLAY_STARE_DART : CAT_PLAY_ZIGZAG_RETREAT;
    if (current == CAT_PLAY_STARE_DART) return r == 0 ? CAT_PLAY_OVAL : CAT_PLAY_ZIGZAG_RETREAT;
    return r == 0 ? CAT_PLAY_OVAL : CAT_PLAY_STARE_DART;
}

static void build_engagement_ranked_transition_weights(
    const struct CatPlayState *state,
    float weights[3]) {
    // No hard-coded algorithm bias: transition probabilities are driven only by
    // each algorithm's observed engagement ranking.
    for (int i = 0; i < 3; ++i) {
        weights[i] = clampf_local(state->algorithm_engagement_scores[i], 0.0f, 1.0f);
    }
}

static enum CatPlayAlgorithm pick_engagement_ranked_alternate_algorithm(
    const struct CatPlayState *state) {
    float weights[3] = {0.0f, 0.0f, 0.0f};
    build_engagement_ranked_transition_weights(state, weights);

    // Always alternate away from the current play algorithm.
    weights[state->algorithm] = 0.0f;

    float total = weights[0] + weights[1] + weights[2];
    if (total <= g_tuning.behavior.transition_weight_epsilon) {
        return pick_other_algorithm(state->algorithm);
    }

    float r = random_float_range(0.0f, total);
    if (r < weights[CAT_PLAY_OVAL]) {
        return CAT_PLAY_OVAL;
    }
    r -= weights[CAT_PLAY_OVAL];
    if (r < weights[CAT_PLAY_STARE_DART]) {
        return CAT_PLAY_STARE_DART;
    }
    return CAT_PLAY_ZIGZAG_RETREAT;
}

static cv::Point2f build_oval_target(
    const Yolov5CatTrackInfo &cat,
    float phase,
    int frame_w,
    int frame_h,
    float arc_scale) {
    const float center_x = cat.x + cat.width * 0.5f;
    const float center_y = cat.y + cat.height * 0.5f;
    const float margin_scale = g_tuning.algorithms.oval_margin_scale;
    const float base_rx = clampf_local(cat.width * 0.5f * margin_scale, g_tuning.algorithms.oval_rx_min, g_tuning.algorithms.oval_rx_max);
    const float base_ry = clampf_local(cat.height * 0.5f * margin_scale, g_tuning.algorithms.oval_ry_min, g_tuning.algorithms.oval_ry_max);
    const float rx = clampf_local(base_rx * arc_scale, g_tuning.algorithms.oval_scaled_min, g_tuning.algorithms.oval_scaled_max);
    const float ry = clampf_local(base_ry * arc_scale, g_tuning.algorithms.oval_scaled_min, g_tuning.algorithms.oval_scaled_max);
    return clamp_point_to_frame(cv::Point2f(center_x + rx * cosf(phase), center_y + ry * sinf(phase)), frame_w, frame_h);
}

static void maybe_transition_with_probability(struct CatPlayState *state, int percent) {
    if ((rand() % 100) < percent) {
        state->algorithm = pick_engagement_ranked_alternate_algorithm(state);
    }
}


static void apply_novelty_decay_recovery(struct CatPlayState *state, float dt_sec) {
    // Session-level novelty memory:
    // - stale algorithms slowly lose weight
    // - all algorithms drift toward a neutral baseline over time
    // - currently active algorithm gets a mild fatigue penalty to avoid lock-in
    const float baseline = g_tuning.algorithms.novelty_baseline;
    const float recovery_rate = g_tuning.algorithms.novelty_recovery_rate;
    const float stale_decay_rate = g_tuning.algorithms.novelty_stale_decay_rate;
    const float active_fatigue_rate = g_tuning.algorithms.novelty_active_fatigue_rate;

    for (int i = 0; i < 3; ++i) {
        float score = state->algorithm_engagement_scores[i];
        score += (baseline - score) * recovery_rate * dt_sec;
        score -= ((i == (int)state->algorithm) ? active_fatigue_rate : stale_decay_rate) * dt_sec;
        state->algorithm_engagement_scores[i] = clampf_local(score, g_tuning.algorithms.novelty_min_score, 1.0f);
    }
}


static void update_cat_velocity_signal(struct CatPlayState *state, const cv::Point2f &cat_center, float dt_sec) {
    if (dt_sec <= 1e-5f) {
        return;
    }

    if (!state->velocity_initialized) {
        state->velocity_initialized = 1;
        state->prev_velocity_cat_center = cat_center;
        state->cat_speed_px_per_sec_ema = 0.0f;
        state->was_within_catch_radius = 0;
        state->was_near_target_zone = 0;
        state->recent_catch_attempt_score = 0.0f;
        state->recent_catch_streak = 0;
        state->recent_miss_streak = 0;
        state->challenge_ladder_level = 0.0f;
        state->prev_engagement_cat_center = cv::Point2f(0.0f, 0.0f);
        state->prev_engagement_laser_point = cv::Point2f(0.0f, 0.0f);
        state->engagement_motion_initialized = 0;
        state->dwell_near_target_time_sec = 0.0f;
        state->disengaged_time_sec = 0.0f;
        return;
    }

    float inst_speed = point_distance(cat_center, state->prev_velocity_cat_center) / dt_sec;
    state->prev_velocity_cat_center = cat_center;
    state->cat_speed_px_per_sec_ema = g_tuning.engagement.velocity_ema_keep * state->cat_speed_px_per_sec_ema +
                                      g_tuning.engagement.velocity_ema_gain * inst_speed;
}

static void update_engagement_score(
    struct CatPlayState *state,
    const cv::Point2f &cat_center,
    const cv::Point2f &laser,
    const Yolov5CatTrackInfo &cat,
    float behavior_confidence,
    float dt_sec) {
    // Engagement uses richer behavior cues:
    // - distance reaction
    // - heading alignment (cat motion toward laser)
    // - dwell quality near target zone
    // - re-engagement latency after temporary disengagement
    float d = point_distance(cat_center, laser);
    float reaction = fabsf(d - state->prev_cat_laser_dist);
    state->prev_cat_laser_dist = d;

    if (!state->engagement_motion_initialized) {
        state->engagement_motion_initialized = 1;
        state->prev_engagement_cat_center = cat_center;
        state->prev_engagement_laser_point = laser;
    }

    const cv::Point2f cat_motion = cat_center - state->prev_engagement_cat_center;
    const float motion_mag = point_distance(cat_center, state->prev_engagement_cat_center);
    const cv::Point2f laser_vec_prev = state->prev_engagement_laser_point - state->prev_engagement_cat_center;
    const float laser_vec_mag = point_distance(state->prev_engagement_laser_point, state->prev_engagement_cat_center);
    float heading_alignment = 0.5f;
    if (motion_mag > 1e-3f && laser_vec_mag > 1e-3f) {
        const float dot = cat_motion.x * laser_vec_prev.x + cat_motion.y * laser_vec_prev.y;
        const float cos_a = clampf_local(dot / (motion_mag * laser_vec_mag), -1.0f, 1.0f);
        heading_alignment = 0.5f * (cos_a + 1.0f);
    }

    state->prev_engagement_cat_center = cat_center;
    state->prev_engagement_laser_point = laser;

    float signal = clampf_local(reaction / g_tuning.engagement.reaction_norm_divisor, 0.0f, 1.0f);
    signal = clampf_local(signal + g_tuning.engagement.heading_alignment_gain * heading_alignment, 0.0f, 1.0f);

    // Boost engagement when cat goes for the dot while this algorithm is active.
    const float catch_radius = compute_catch_radius(cat, d, behavior_confidence);
    const float near_target_radius = g_tuning.engagement.near_target_radius_scale * catch_radius;
    const int within_catch_radius = (d <= catch_radius) ? 1 : 0;
    const int near_target_zone = (d <= near_target_radius) ? 1 : 0;
    const int catch_enter_event = (within_catch_radius && !state->was_within_catch_radius) ? 1 : 0;
    const int near_enter_event = (near_target_zone && !state->was_near_target_zone) ? 1 : 0;
    state->was_near_target_zone = near_target_zone;
    state->was_within_catch_radius = within_catch_radius;

    if (near_target_zone) {
        state->dwell_near_target_time_sec = clampf_local(state->dwell_near_target_time_sec + dt_sec, 0.0f, g_tuning.engagement.dwell_max_sec);
        signal = clampf_local(signal + g_tuning.engagement.dwell_gain * clampf_local(state->dwell_near_target_time_sec / g_tuning.engagement.dwell_ramp_sec, 0.0f, 1.0f), 0.0f, 1.0f);
    } else {
        state->dwell_near_target_time_sec = clampf_local(state->dwell_near_target_time_sec - g_tuning.engagement.dwell_decay_rate * dt_sec, 0.0f, g_tuning.engagement.dwell_max_sec);
    }

    if (within_catch_radius) {
        signal = clampf_local(signal + g_tuning.engagement.catch_bonus, 0.0f, 1.0f);
    }

    if (!near_target_zone) {
        state->disengaged_time_sec = clampf_local(state->disengaged_time_sec + dt_sec, 0.0f, g_tuning.engagement.disengaged_max_sec);
    } else {
        if (near_enter_event && state->disengaged_time_sec > g_tuning.engagement.reengage_min_disengaged_sec) {
            const float latency_bonus = 1.0f - clampf_local(state->disengaged_time_sec / g_tuning.engagement.reengage_norm_sec, 0.0f, 1.0f);
            signal = clampf_local(signal + g_tuning.engagement.reengage_bonus_gain * latency_bonus, 0.0f, 1.0f);
        }
        state->disengaged_time_sec = 0.0f;
    }

    // Track recent successful catch attempts (distance entering catch radius)
    // to adapt near-miss burst/pause parameters.
    state->recent_catch_attempt_score =
        clampf_local(g_tuning.engagement.catch_attempt_decay * state->recent_catch_attempt_score +
                         g_tuning.engagement.catch_attempt_gain * (float)catch_enter_event,
                     0.0f,
                     1.0f);

    if (catch_enter_event) {
        state->recent_catch_streak = (state->recent_catch_streak < 10) ? (state->recent_catch_streak + 1) : 10;
        state->recent_miss_streak = (state->recent_miss_streak > 0) ? (state->recent_miss_streak - 1) : 0;
    } else if (near_enter_event && !within_catch_radius) {
        state->recent_miss_streak = (state->recent_miss_streak < 12) ? (state->recent_miss_streak + 1) : 12;
        state->recent_catch_streak = (state->recent_catch_streak > 0) ? (state->recent_catch_streak - 1) : 0;
    } else {
        state->recent_catch_streak = (state->recent_catch_streak > 0) ? (state->recent_catch_streak - 1) : 0;
        state->recent_miss_streak = (state->recent_miss_streak > 0) ? (state->recent_miss_streak - 1) : 0;
    }

    const float ladder_target = clampf_local(
        g_tuning.engagement.catch_challenge_weight * (float)state->recent_catch_streak -
            g_tuning.engagement.near_target_challenge_weight * (float)state->recent_miss_streak,
        g_tuning.behavior.near_target_challenge_min,
        g_tuning.behavior.near_target_challenge_max);
    state->challenge_ladder_level =
        clampf_local(g_tuning.engagement.challenge_ema_keep * state->challenge_ladder_level +
                         g_tuning.engagement.challenge_ema_gain * ladder_target,
                     g_tuning.behavior.near_target_challenge_min,
                     g_tuning.behavior.near_target_challenge_max);

    state->engagement_score = g_tuning.engagement.score_ema_keep * state->engagement_score + g_tuning.engagement.score_ema_gain * signal;
    const int algo_index = (int)state->algorithm;
    state->algorithm_engagement_scores[algo_index] =
        g_tuning.engagement.score_ema_keep * state->algorithm_engagement_scores[algo_index] +
        g_tuning.engagement.score_ema_gain * signal;
}



static enum PlayDirectorIntent choose_next_director_intent(const struct CatPlayState *state) {
    const float speed_norm = clampf_local(state->cat_speed_px_per_sec_ema / g_tuning.algorithms.speed_norm_divisor, 0.0f, 1.0f);
    const float engage = clampf_local(state->engagement_score, 0.0f, 1.0f);
    const float catch_success = clampf_local(state->recent_catch_attempt_score, 0.0f, 1.0f);

    float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    w[DIRECTOR_INTENT_TEASE] = g_tuning.director.director_w_tease_base +
                               g_tuning.director.director_w_tease_low_engage_gain * (1.0f - engage);
    w[DIRECTOR_INTENT_CHASE] = g_tuning.director.director_w_chase_base +
                               g_tuning.director.director_w_chase_speed_gain * speed_norm;
    w[DIRECTOR_INTENT_POUNCE_WINDOW] = g_tuning.director.director_w_pounce_base +
                                       g_tuning.director.director_w_pounce_catch_gain * catch_success +
                                       g_tuning.director.director_w_pounce_low_speed_gain * (1.0f - speed_norm);
    w[DIRECTOR_INTENT_RECOVER] = g_tuning.director.director_w_recover_base +
                                 g_tuning.director.director_w_recover_speed_minus_engage_gain *
                                     clampf_local(speed_norm - engage, 0.0f, 1.0f);

    float total = w[0] + w[1] + w[2] + w[3];
    if (total <= 1e-5f) {
        return DIRECTOR_INTENT_TEASE;
    }

    float r = random_float_range(0.0f, total);
    for (int i = 0; i < 4; ++i) {
        if (r < w[i]) {
            return (enum PlayDirectorIntent)i;
        }
        r -= w[i];
    }
    return DIRECTOR_INTENT_RECOVER;
}

static void update_director_layer(struct CatPlayState *state, float dt_sec) {
    state->director_time_remaining_sec -= dt_sec;
    if (state->director_time_remaining_sec > 0.0f) {
        return;
    }

    state->director_intent = choose_next_director_intent(state);
    state->director_time_remaining_sec =
        random_float_range(g_tuning.director.director_duration_min_sec, g_tuning.director.director_duration_max_sec);
}
static void maybe_start_near_miss_tease(
    struct CatPlayState *state,
    const Yolov5CatTrackInfo &cat,
    float dt_sec,
    float director_tease_bias) {
    if (state->near_miss_phase != NEAR_MISS_OFF) {
        return;
    }

    // Trigger occasionally during oval play; stronger chance when engagement is low.
    const float low_engagement_boost = clampf_local(
        g_tuning.algorithms.near_miss_low_engagement_threshold - state->engagement_score,
        0.0f,
        g_tuning.algorithms.near_miss_low_engagement_threshold);
    const float trigger_prob_per_sec = (g_tuning.algorithms.near_miss_trigger_base + low_engagement_boost) * director_tease_bias;
    if (random_float_range(0.0f, 1.0f) > trigger_prob_per_sec * clampf_local(dt_sec, 0.0f, 1.0f)) {
        return;
    }

    state->near_miss_phase = NEAR_MISS_BURST;
    const float catch_success = clampf_local(state->recent_catch_attempt_score, 0.0f, 1.0f);
    const float challenge = clampf_local(state->challenge_ladder_level, -1.0f, 1.0f);
    const int pass_bias = (int)floorf(g_tuning.behavior.rounding_bias +
                                      g_tuning.algorithms.near_miss_pass_bias_scale * catch_success); // high success => longer tease runs.
    state->near_miss_passes_remaining = 2 + (rand() % 3) + pass_bias; // adaptive ~2-7 passes.
    state->near_miss_angle_rad = random_float_range(0.0f, 6.2831853f);
    const float radius_min = g_tuning.algorithms.near_miss_radius_min -
                             g_tuning.algorithms.near_miss_radius_challenge_min_gain * challenge;
    const float radius_max = g_tuning.algorithms.near_miss_radius_max -
                             g_tuning.algorithms.near_miss_radius_challenge_max_gain * challenge;
    state->near_miss_radius_scale = random_float_range(radius_min, radius_max);
    state->near_miss_segment_time_sec = 0.0f;
    state->near_miss_segment_duration_sec =
        random_float_range(g_tuning.algorithms.near_miss_segment_min_sec, g_tuning.algorithms.near_miss_segment_max_sec);
    state->near_miss_pause_time_sec = 0.0f;
    state->near_miss_direction = (rand() % 2 == 0) ? 1 : -1;
    (void)cat;
}

static int maybe_build_near_miss_tease_target(
    struct CatPlayState *state,
    const Yolov5CatTrackInfo &cat,
    float behavior_confidence,
    const cv::Point2f &cat_center,
    const cv::Point2f &laser,
    int frame_w,
    int frame_h,
    float dt_sec,
    const char **algo_name_out,
    cv::Point2f *target_out) {
    if (state->near_miss_phase == NEAR_MISS_OFF) {
        return 0;
    }

    if (state->near_miss_phase == NEAR_MISS_PAUSE) {
        state->near_miss_pause_time_sec -= dt_sec;
        *algo_name_out = "near_miss_tease_pause";
        *target_out = clamp_point_to_frame(state->near_miss_pause_point, frame_w, frame_h);
        if (state->near_miss_pause_time_sec <= 0.0f) {
            state->near_miss_phase = NEAR_MISS_OFF;
        }
        return 1;
    }

    const float radius = compute_catch_radius(cat, point_distance(cat_center, laser), behavior_confidence) * state->near_miss_radius_scale;
    state->near_miss_angle_rad += (float)state->near_miss_direction * dt_sec * g_tuning.algorithms.near_miss_orbit_speed;
    cv::Point2f tease_point(
        cat_center.x + cosf(state->near_miss_angle_rad) * radius,
        cat_center.y + sinf(state->near_miss_angle_rad) * radius);
    tease_point = clamp_point_to_frame(tease_point, frame_w, frame_h);

    state->near_miss_segment_time_sec += dt_sec;
    if (state->near_miss_segment_time_sec >= state->near_miss_segment_duration_sec) {
        state->near_miss_segment_time_sec = 0.0f;
        state->near_miss_segment_duration_sec =
            random_float_range(g_tuning.algorithms.near_miss_segment_min_sec, g_tuning.algorithms.near_miss_segment_max_sec);
        state->near_miss_passes_remaining--;
        if ((rand() % 100) < (int)g_tuning.algorithms.near_miss_direction_flip_percent) {
            state->near_miss_direction = -state->near_miss_direction;
        }
    }

    if (state->near_miss_passes_remaining <= 0) {
        state->near_miss_phase = NEAR_MISS_PAUSE;
        const float catch_success = clampf_local(state->recent_catch_attempt_score, 0.0f, 1.0f);
        const float challenge = clampf_local(state->challenge_ladder_level, -1.0f, 1.0f);
        const float pause_min =
            (g_tuning.algorithms.near_miss_pause_min_base +
             g_tuning.algorithms.near_miss_pause_min_gain * (1.0f - catch_success)) *
            (1.0f - g_tuning.algorithms.near_miss_pause_challenge_min_gain * challenge);
        const float pause_max =
            (g_tuning.algorithms.near_miss_pause_max_base +
             g_tuning.algorithms.near_miss_pause_max_gain * (1.0f - catch_success)) *
            (1.0f - g_tuning.algorithms.near_miss_pause_challenge_max_gain * challenge);
        state->near_miss_pause_time_sec = random_float_range(pause_min, pause_max);
        state->near_miss_pause_point = tease_point;
        *algo_name_out = "near_miss_tease_pause";
        *target_out = tease_point;
        return 1;
    }

    *algo_name_out = "near_miss_tease";
    *target_out = tease_point;
    return 1;
}

static void maybe_flip_oval_direction_on_catch(struct CatPlayState *state,
                                               const Yolov5CatTrackInfo &cat,
                                               float behavior_confidence,
                                               const cv::Point2f &cat_center,
                                               const cv::Point2f &laser,
                                               float dt_sec) {
    state->oval_direction_cooldown_sec =
        clampf_local(state->oval_direction_cooldown_sec - dt_sec, 0.0f, g_tuning.hesitation.cooldown_max_clamp);

    // "Trying to catch" heuristic: laser gets very close to cat center.
    const float catch_radius = compute_catch_radius(cat, point_distance(cat_center, laser), behavior_confidence);
    if (point_distance(cat_center, laser) > catch_radius || state->oval_direction_cooldown_sec > 0.0f) {
        return;
    }

    // Randomized reaction on catch: each catch event gets its own flip probability,
    // then we sample whether orbit direction changes.
    const float flip_chance = random_float_range(g_tuning.algorithms.oval_flip_chance_min, g_tuning.algorithms.oval_flip_chance_max);
    if (random_float_range(0.0f, 1.0f) < flip_chance) {
        state->oval_direction = -state->oval_direction;
    }

    // Debounce to avoid flipping every frame while the cat hovers near the laser.
    state->oval_direction_cooldown_sec =
        random_float_range(g_tuning.algorithms.oval_flip_cooldown_min_sec, g_tuning.algorithms.oval_flip_cooldown_max_sec);
}

void init_cat_play_state(struct CatPlayState *state) {
    ensure_tuning_initialized();
    state->algorithm = CAT_PLAY_OVAL;
    state->last_cat_center = cv::Point2f(0.0f, 0.0f);
    state->cat_still_time_sec = 0.0f;
    state->velocity_initialized = 0;
    state->prev_velocity_cat_center = cv::Point2f(0.0f, 0.0f);
    state->cat_speed_px_per_sec_ema = 0.0f;
    state->was_within_catch_radius = 0;
    state->was_near_target_zone = 0;
    state->recent_catch_attempt_score = 0.0f;
    state->recent_catch_streak = 0;
    state->recent_miss_streak = 0;
    state->challenge_ladder_level = 0.0f;
    state->prev_engagement_cat_center = cv::Point2f(0.0f, 0.0f);
    state->prev_engagement_laser_point = cv::Point2f(0.0f, 0.0f);
    state->engagement_motion_initialized = 0;
    state->dwell_near_target_time_sec = 0.0f;
    state->disengaged_time_sec = 0.0f;
    state->engagement_score = 0.0f;
    state->algorithm_engagement_scores[CAT_PLAY_OVAL] = 1.0f;
    state->algorithm_engagement_scores[CAT_PLAY_STARE_DART] = 1.0f;
    state->algorithm_engagement_scores[CAT_PLAY_ZIGZAG_RETREAT] = 1.0f;
    state->prev_cat_laser_dist = 0.0f;
    state->session_time_sec = 0.0f;
    state->calm_time_sec = 0.0f;
    state->close_chase_time_sec = 0.0f;
    state->hesitation_pause_time_sec = 0.0f;
    state->hesitation_cooldown_sec = 0.0f;
    state->director_intent = DIRECTOR_INTENT_TEASE;
    state->director_time_remaining_sec =
        random_float_range(g_tuning.director.director_duration_min_sec, g_tuning.director.director_duration_max_sec);
    state->oval_phase = 0.0f;
    state->oval_direction = 1;
    state->oval_direction_cooldown_sec = 0.0f;
    state->near_miss_phase = NEAR_MISS_OFF;
    state->near_miss_passes_remaining = 0;
    state->near_miss_angle_rad = 0.0f;
    state->near_miss_radius_scale = g_tuning.algorithms.near_miss_initial_radius_scale;
    state->near_miss_segment_time_sec = 0.0f;
    state->near_miss_segment_duration_sec = 0.0f;
    state->near_miss_pause_time_sec = 0.0f;
    state->near_miss_direction = 1;
    state->near_miss_pause_point = cv::Point2f(0.0f, 0.0f);
    state->stare_dart_phase = STARE_DART_HOLD;
    state->stare_dart_hold_time_sec = 0.0f;
    state->stare_dart_hold_point = cv::Point2f(0.0f, 0.0f);
    state->stare_dart_dart_target = cv::Point2f(0.0f, 0.0f);
    state->zigzag_phase = ZIGZAG_APPROACH;
    state->zigzag_phase_time_sec = 0.0f;
    state->zigzag_front_point = cv::Point2f(0.0f, 0.0f);
    state->zigzag_retreat_point = cv::Point2f(0.0f, 0.0f);
}

cv::Point2f build_cat_play_target(
    struct CatPlayState *state,
    const Yolov5CatTrackInfo &cat,
    float detection_confidence,
    const cv::Point2f &laser,
    int frame_index,
    int frame_w,
    int frame_h,
    float dt_sec,
    const char **algo_name_out) {
    ensure_tuning_initialized();
    (void)frame_index;
    // Gameplay intent map:
    // - OVAL: baseline orbit/tease motion around cat (primary default behavior).
    // - STARE_DART: hold still to lure focus, then dart to trigger pounce.
    // - ZIGZAG_RETREAT: high-arousal chase micro-bursts with retreat windows.
    //
    // Director intent and engagement/challenge scores modulate these patterns,
    // while detection confidence adds a fallback style layer (calmer when low).
    const float behavior_confidence = clampf_local(detection_confidence, 0.0f, 1.0f);
    const float confidence_norm = clampf_local((behavior_confidence - 0.30f) / 0.60f, 0.0f, 1.0f);
    const float low_confidence = 1.0f - confidence_norm;
    const float confidence_speed_scale =
        g_tuning.confidence_fallback.speed_scale_min + g_tuning.confidence_fallback.speed_scale_gain * confidence_norm;
    const float confidence_arc_scale = 1.0f + g_tuning.confidence_fallback.arc_scale_gain * low_confidence;
    const float confidence_hold_scale = 1.0f + g_tuning.confidence_fallback.hold_scale_gain * low_confidence;
    const float confidence_dart_scale =
        g_tuning.confidence_fallback.dart_scale_min + g_tuning.confidence_fallback.dart_scale_gain * confidence_norm;
    const float confidence_transition_scale =
        g_tuning.confidence_fallback.transition_scale_base +
        g_tuning.confidence_fallback.transition_scale_gain * confidence_norm;
    const cv::Point2f cat_center(cat.x + cat.width * 0.5f, cat.y + cat.height * 0.5f);

    update_engagement_score(state, cat_center, laser, cat, behavior_confidence, dt_sec);
    update_cat_velocity_signal(state, cat_center, dt_sec);
    apply_novelty_decay_recovery(state, dt_sec);
    update_director_layer(state, dt_sec);

    // Anti-fatigue: every ~120s of active play, insert a short calm interval.
    state->session_time_sec += dt_sec;
    if (state->session_time_sec >= g_tuning.behavior.session_calm_trigger_sec && state->calm_time_sec <= 0.0f) {
        state->calm_time_sec = random_float_range(g_tuning.behavior.calm_pause_min_sec, g_tuning.behavior.calm_pause_max_sec);
        state->session_time_sec = 0.0f;
    }
    if (state->calm_time_sec > 0.0f) {
        *algo_name_out = "calm_pause";
        state->calm_time_sec -= dt_sec;
        return laser;
    }

    if (state->hesitation_cooldown_sec > 0.0f) {
        state->hesitation_cooldown_sec =
            clampf_local(state->hesitation_cooldown_sec - dt_sec, 0.0f, g_tuning.hesitation.cooldown_max_clamp);
    }

    if (state->hesitation_pause_time_sec > 0.0f) {
        *algo_name_out = "hesitation_pause";
        state->hesitation_pause_time_sec -= dt_sec;
        return laser;
    }

    const float catch_radius = compute_catch_radius(cat, point_distance(cat_center, laser), behavior_confidence);
    const float speed_norm_for_hesitation = clampf_local(state->cat_speed_px_per_sec_ema / g_tuning.algorithms.speed_norm_divisor, 0.0f, 1.0f);
    const float chase_dist = point_distance(cat_center, laser);
    const int high_speed_close_chase =
        (speed_norm_for_hesitation > g_tuning.hesitation.close_chase_speed_threshold &&
         chase_dist < (g_tuning.hesitation.close_chase_dist_scale * catch_radius));

    if (high_speed_close_chase) {
        state->close_chase_time_sec += dt_sec;
    } else {
        state->close_chase_time_sec = clampf_local(
            state->close_chase_time_sec - dt_sec * g_tuning.hesitation.close_chase_decay_rate,
            0.0f,
            g_tuning.hesitation.close_chase_max_sec);
    }

    if (state->hesitation_cooldown_sec <= 0.0f && state->close_chase_time_sec >= g_tuning.hesitation.close_chase_trigger_sec) {
        state->hesitation_pause_time_sec = random_float_range(g_tuning.hesitation.pause_min_sec, g_tuning.hesitation.pause_max_sec);
        state->hesitation_cooldown_sec = random_float_range(g_tuning.hesitation.cooldown_min_sec, g_tuning.hesitation.cooldown_max_sec);
        state->close_chase_time_sec = 0.0f;
        *algo_name_out = "hesitation_pause";
        return laser;
    }

    if (point_distance(cat_center, state->last_cat_center) < g_tuning.behavior.still_distance_px) state->cat_still_time_sec += dt_sec;
    else {
        state->cat_still_time_sec = 0.0f;
        state->last_cat_center = cat_center;
    }
    if (state->cat_still_time_sec >= g_tuning.behavior.still_switch_sec) {
        state->algorithm = pick_engagement_ranked_alternate_algorithm(state);
        state->cat_still_time_sec = 0.0f;
    }

    float director_tease_bias = 1.0f;
    float oval_speed_scale = 1.0f;
    float zigzag_speed_scale = 1.0f;
    float zigzag_duration_scale = 1.0f;
    if (state->director_intent == DIRECTOR_INTENT_TEASE) {
        director_tease_bias = g_tuning.director.tease_director_bias;
        if (state->algorithm != CAT_PLAY_OVAL && random_float_range(0.0f, 1.0f) < g_tuning.director.tease_to_oval_switch_prob) {
            state->algorithm = CAT_PLAY_OVAL;
        }
    } else if (state->director_intent == DIRECTOR_INTENT_CHASE) {
        director_tease_bias = g_tuning.director.chase_director_bias;
        zigzag_speed_scale = g_tuning.director.chase_zigzag_speed_scale;
        zigzag_duration_scale = g_tuning.director.chase_zigzag_duration_scale;
        if (state->algorithm == CAT_PLAY_OVAL && random_float_range(0.0f, 1.0f) < g_tuning.director.chase_oval_to_zigzag_switch_prob) {
            state->algorithm = CAT_PLAY_ZIGZAG_RETREAT;
            state->zigzag_phase = ZIGZAG_APPROACH;
        }
    } else if (state->director_intent == DIRECTOR_INTENT_POUNCE_WINDOW) {
        director_tease_bias = g_tuning.director.pounce_director_bias;
        oval_speed_scale = g_tuning.director.pounce_oval_speed_scale;
        if (state->algorithm != CAT_PLAY_OVAL && random_float_range(0.0f, 1.0f) < g_tuning.director.pounce_to_oval_switch_prob) {
            state->algorithm = CAT_PLAY_OVAL;
        }
    } else { // DIRECTOR_INTENT_RECOVER
        director_tease_bias = g_tuning.director.recover_director_bias;
        oval_speed_scale = g_tuning.director.recover_oval_speed_scale;
        zigzag_speed_scale = g_tuning.director.recover_zigzag_speed_scale;
        zigzag_duration_scale = g_tuning.director.recover_zigzag_duration_scale;
    }
    oval_speed_scale *= confidence_speed_scale;
    zigzag_speed_scale *= confidence_speed_scale;
    zigzag_duration_scale *= (1.0f + g_tuning.confidence_fallback.zigzag_duration_low_conf_gain * low_confidence);

    if (state->algorithm == CAT_PLAY_OVAL) {
        // OVAL branch: smooth, circular prey motion with optional near-miss tease inserts.
        maybe_flip_oval_direction_on_catch(state, cat, behavior_confidence, cat_center, laser, dt_sec);
        maybe_start_near_miss_tease(state, cat, dt_sec, director_tease_bias);

        cv::Point2f tease_target(0.0f, 0.0f);
        if (maybe_build_near_miss_tease_target(state, cat, behavior_confidence, cat_center, laser, frame_w, frame_h, dt_sec, algo_name_out, &tease_target)) {
            return tease_target;
        }

        *algo_name_out = "oval";
        const float speed_norm = clampf_local(state->cat_speed_px_per_sec_ema / g_tuning.algorithms.speed_norm_divisor, 0.0f, 1.0f);
        const float challenge = clampf_local(state->challenge_ladder_level, -1.0f, 1.0f);
        const float direction = (state->oval_direction >= 0) ? 1.0f : -1.0f;
        const float oval_challenge_speed =
            clampf_local(1.0f + g_tuning.behavior.oval_challenge_speed_gain * challenge,
                         g_tuning.behavior.oval_challenge_speed_min,
                         g_tuning.behavior.oval_challenge_speed_max);
        const float oval_phase_step =
            (g_tuning.behavior.oval_phase_base_step + g_tuning.behavior.oval_phase_speed_gain * speed_norm) *
            oval_speed_scale * oval_challenge_speed;
        const float oval_arc_scale = clampf_local(
            (g_tuning.behavior.oval_challenge_arc_base - g_tuning.behavior.oval_challenge_arc_gain * challenge) *
                confidence_arc_scale,
            g_tuning.behavior.oval_challenge_arc_min,
            g_tuning.behavior.oval_challenge_arc_max);
        state->oval_phase += direction * oval_phase_step;

        // If cat movement is low, occasionally bait with a dart pattern.
        if (speed_norm < g_tuning.stare_dart.low_speed_threshold &&
            random_float_range(0.0f, 1.0f) <
                (g_tuning.stare_dart.low_speed_dart_prob * confidence_dart_scale *
                 clampf_local(dt_sec * g_tuning.behavior.random_gate_scale,
                              g_tuning.behavior.random_gate_min,
                              g_tuning.behavior.random_gate_max))) {
            state->algorithm = CAT_PLAY_STARE_DART;
            state->stare_dart_phase = STARE_DART_HOLD;
            state->stare_dart_hold_time_sec = random_float_range(g_tuning.stare_dart.hold_min_sec, g_tuning.stare_dart.hold_max_sec) * confidence_hold_scale;
            state->stare_dart_hold_point = laser;
        }
        return build_oval_target(cat, state->oval_phase, frame_w, frame_h, oval_arc_scale);
    }

    if (state->algorithm == CAT_PLAY_STARE_DART) {
        // STARE_DART branch: alternate between stillness (attention anchor)
        // and sudden relocation (prey escape cue).
        *algo_name_out = "stare_dart";
        if (state->stare_dart_phase == STARE_DART_HOLD) {
            if (state->stare_dart_hold_time_sec <= 0.0f) {
                state->stare_dart_hold_time_sec = random_float_range(g_tuning.stare_dart.long_hold_min_sec, g_tuning.stare_dart.long_hold_max_sec) * confidence_hold_scale;
                state->stare_dart_hold_point = laser;
            }
            state->stare_dart_hold_time_sec -= dt_sec;
            if (state->stare_dart_hold_time_sec <= 0.0f) {
                state->stare_dart_phase = STARE_DART_DART;
                state->stare_dart_dart_target = random_point_outside_cat(cat, frame_w, frame_h);
                const int transition_percent =
                    (int)floorf(g_tuning.behavior.rounding_bias +
                                g_tuning.behavior.transition_base_percent * confidence_transition_scale);
                maybe_transition_with_probability(state, transition_percent);
            }
            return clamp_point_to_frame(state->stare_dart_hold_point, frame_w, frame_h);
        }

        if (point_in_cat_bbox(state->stare_dart_dart_target, cat)) {
            state->stare_dart_dart_target = random_point_outside_cat(cat, frame_w, frame_h);
        }
        if (point_distance(laser, state->stare_dart_dart_target) < g_tuning.zigzag.retreat_reach_threshold_px) {
            state->stare_dart_phase = STARE_DART_HOLD;
            state->stare_dart_hold_time_sec = random_float_range(g_tuning.stare_dart.long_hold_min_sec, g_tuning.stare_dart.long_hold_max_sec) * confidence_hold_scale;
            state->stare_dart_hold_point = laser;
        }
        return clamp_point_to_frame(state->stare_dart_dart_target, frame_w, frame_h);
    }

    *algo_name_out = "zigzag_retreat";
    // ZIGZAG_RETREAT branch: approach, lateral shake near the cat, then retreat.
    // This creates a challenge ladder-friendly burst pattern for chase periods.
    if (state->zigzag_phase == ZIGZAG_APPROACH) {
        state->zigzag_front_point = clamp_point_to_frame(cv::Point2f(cat_center.x, cat.y - cat.height * g_tuning.zigzag.front_y_offset), frame_w, frame_h);
        if (point_distance(laser, state->zigzag_front_point) < g_tuning.zigzag.front_reach_threshold_px) {
            state->zigzag_phase = ZIGZAG_SHAKE;
            state->zigzag_phase_time_sec = 0.0f;
        }
        return state->zigzag_front_point;
    }

    if (state->zigzag_phase == ZIGZAG_SHAKE) {
        state->zigzag_phase_time_sec += dt_sec;
        const float challenge = clampf_local(state->challenge_ladder_level, -1.0f, 1.0f);
        const float zigzag_arc_scale =
            clampf_local((g_tuning.zigzag.challenge_arc_base - g_tuning.zigzag.challenge_arc_gain * challenge) *
                             confidence_arc_scale,
                         g_tuning.zigzag.arc_scale_min,
                         g_tuning.zigzag.arc_scale_max);
        const float amp_x = clampf_local(
            clampf_local(cat.width * g_tuning.zigzag.amp_x_scale, g_tuning.zigzag.amp_x_min, g_tuning.zigzag.amp_x_max) * zigzag_arc_scale,
            g_tuning.zigzag.amp_x_scaled_min,
            g_tuning.zigzag.amp_x_scaled_max);
        const float amp_y = clampf_local(
            clampf_local(cat.height * g_tuning.zigzag.amp_y_scale, g_tuning.zigzag.amp_y_min, g_tuning.zigzag.amp_y_max) * zigzag_arc_scale,
            g_tuning.zigzag.amp_y_scaled_min,
            g_tuning.zigzag.amp_y_scaled_max);
        const float speed_norm = clampf_local(state->cat_speed_px_per_sec_ema / g_tuning.algorithms.speed_norm_divisor, 0.0f, 1.0f);
        const float zigzag_challenge_speed =
            clampf_local(1.0f + g_tuning.zigzag.challenge_speed_gain * challenge,
                         g_tuning.zigzag.challenge_speed_min,
                         g_tuning.zigzag.challenge_speed_max);
        const float t = state->zigzag_phase_time_sec *
                        (g_tuning.zigzag.wave_speed_base + g_tuning.zigzag.wave_speed_gain * speed_norm) *
                        zigzag_speed_scale * zigzag_challenge_speed;
        const float saw = (fmodf(t, 2.0f) < 1.0f) ? 1.0f : -1.0f;
        cv::Point2f p(state->zigzag_front_point.x + saw * amp_x,
                      state->zigzag_front_point.y + sinf(t * g_tuning.zigzag.wave_y_freq) * amp_y);
        const float shake_duration_sec =
            (g_tuning.zigzag.shake_duration_base - g_tuning.zigzag.shake_duration_speed_gain * speed_norm) *
            zigzag_duration_scale *
            clampf_local(1.0f - g_tuning.zigzag.challenge_shake_duration_gain * challenge,
                         g_tuning.zigzag.challenge_shake_duration_min,
                         g_tuning.zigzag.challenge_shake_duration_max);
        if (state->zigzag_phase_time_sec >= shake_duration_sec) {
            state->zigzag_phase = ZIGZAG_RETREAT;
            state->zigzag_retreat_point = furthest_frame_corner_from_point(laser, frame_w, frame_h);
        }
        return clamp_point_to_frame(p, frame_w, frame_h);
    }

    if (state->zigzag_phase == ZIGZAG_RETREAT) {
        state->zigzag_retreat_point = build_retreat_target_avoiding_cat(laser, cat, frame_w, frame_h);
        if (point_distance(laser, state->zigzag_retreat_point) < g_tuning.zigzag.retreat_reach_threshold_px) {
            state->zigzag_phase = ZIGZAG_RETURN;
            const int transition_percent =
                (int)floorf(g_tuning.behavior.rounding_bias +
                            g_tuning.behavior.transition_base_percent * confidence_transition_scale);
            maybe_transition_with_probability(state, transition_percent);
        }
        return state->zigzag_retreat_point;
    }

    state->zigzag_front_point = clamp_point_to_frame(cv::Point2f(cat_center.x, cat.y - cat.height * g_tuning.zigzag.front_y_offset), frame_w, frame_h);
    if (point_distance(laser, state->zigzag_front_point) < g_tuning.zigzag.front_reach_threshold_px) {
        state->zigzag_phase = ZIGZAG_SHAKE;
        state->zigzag_phase_time_sec = 0.0f;
    }
    return state->zigzag_front_point;
}
