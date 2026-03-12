#include "play_algorithms.h"
#include "play_algorithms_internal.h"
#include "play_director_policy.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>

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
    update_director_layer_policy(state, dt_sec);

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
