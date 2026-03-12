#include "play_algorithms.h"
#include "play_algorithms_internal.h"

#include <opencv2/imgproc.hpp>

CatPlayTuningConfig g_tuning;
int g_tuning_initialized = 0;

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

void ensure_tuning_initialized(void) {
    if (!g_tuning_initialized) {
        reset_cat_play_tuning_defaults();
    }
}
