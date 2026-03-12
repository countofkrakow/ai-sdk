#include "play_algorithms.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>

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
    return clampf_local(0.18f * (cat.width + cat.height), 10.0f, 42.0f);
}

static float compute_catch_radius(const Yolov5CatTrackInfo &cat, float cat_laser_dist) {
    const float base = compute_base_catch_radius(cat);
    const float bbox_diag = sqrtf(cat.width * cat.width + cat.height * cat.height);

    // Lower confidence and larger cat-dot distance narrow the effective catch zone,
    // while high confidence and close-range play can widen it for challenge.
    const float confidence_norm = clampf_local((cat.confidence - 0.25f) / 0.60f, 0.0f, 1.0f);
    const float confidence_scale = 0.90f + 0.25f * confidence_norm;

    const float near_dist = 1.1f * bbox_diag;
    const float far_dist = 2.2f * bbox_diag + 1.0f;
    const float close_norm = 1.0f - clampf_local((cat_laser_dist - near_dist) / (far_dist - near_dist), 0.0f, 1.0f);
    const float distance_scale = 0.88f + 0.20f * close_norm;

    return clampf_local(base * confidence_scale * distance_scale, 8.0f, 52.0f);
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

    const float clearance = 24.0f;
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
    if (total <= 0.0001f) {
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

static cv::Point2f build_oval_target(const Yolov5CatTrackInfo &cat, float phase, int frame_w, int frame_h) {
    const float center_x = cat.x + cat.width * 0.5f;
    const float center_y = cat.y + cat.height * 0.5f;
    const float margin_scale = 1.15f;
    const float rx = clampf_local(cat.width * 0.5f * margin_scale, 12.0f, 140.0f);
    const float ry = clampf_local(cat.height * 0.5f * margin_scale, 12.0f, 140.0f);
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
    const float baseline = 0.50f;
    const float recovery_rate = 0.04f;
    const float stale_decay_rate = 0.012f;
    const float active_fatigue_rate = 0.02f;

    for (int i = 0; i < 3; ++i) {
        float score = state->algorithm_engagement_scores[i];
        score += (baseline - score) * recovery_rate * dt_sec;
        score -= ((i == (int)state->algorithm) ? active_fatigue_rate : stale_decay_rate) * dt_sec;
        state->algorithm_engagement_scores[i] = clampf_local(score, 0.05f, 1.0f);
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
    state->recent_catch_attempt_score = 0.0f;
        return;
    }

    float inst_speed = point_distance(cat_center, state->prev_velocity_cat_center) / dt_sec;
    state->prev_velocity_cat_center = cat_center;
    state->cat_speed_px_per_sec_ema = 0.88f * state->cat_speed_px_per_sec_ema + 0.12f * inst_speed;
}

static void update_engagement_score(struct CatPlayState *state, const cv::Point2f &cat_center, const cv::Point2f &laser, const Yolov5CatTrackInfo &cat) {
    // Heuristic: engagement increases when cat-laser distance changes (cat reacts).
    float d = point_distance(cat_center, laser);
    float reaction = fabsf(d - state->prev_cat_laser_dist);
    state->prev_cat_laser_dist = d;
    float signal = clampf_local(reaction / 25.0f, 0.0f, 1.0f);

    // Boost engagement when cat goes for the dot while this algorithm is active.
    const float catch_radius = compute_catch_radius(cat, d);
    const int within_catch_radius = (d <= catch_radius) ? 1 : 0;
    const int catch_enter_event = (within_catch_radius && !state->was_within_catch_radius) ? 1 : 0;
    state->was_within_catch_radius = within_catch_radius;

    if (within_catch_radius) {
        signal = clampf_local(signal + 0.35f, 0.0f, 1.0f);
    }

    // Track recent successful catch attempts (distance entering catch radius)
    // to adapt near-miss burst/pause parameters.
    state->recent_catch_attempt_score =
        clampf_local(0.92f * state->recent_catch_attempt_score + 0.08f * (float)catch_enter_event, 0.0f, 1.0f);

    state->engagement_score = 0.9f * state->engagement_score + 0.1f * signal;
    const int algo_index = (int)state->algorithm;
    state->algorithm_engagement_scores[algo_index] =
        0.9f * state->algorithm_engagement_scores[algo_index] + 0.1f * signal;
}



static enum PlayDirectorIntent choose_next_director_intent(const struct CatPlayState *state) {
    const float speed_norm = clampf_local(state->cat_speed_px_per_sec_ema / 220.0f, 0.0f, 1.0f);
    const float engage = clampf_local(state->engagement_score, 0.0f, 1.0f);
    const float catch_success = clampf_local(state->recent_catch_attempt_score, 0.0f, 1.0f);

    float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    w[DIRECTOR_INTENT_TEASE] = 0.22f + 0.36f * (1.0f - engage);
    w[DIRECTOR_INTENT_CHASE] = 0.18f + 0.62f * speed_norm;
    w[DIRECTOR_INTENT_POUNCE_WINDOW] = 0.14f + 0.46f * catch_success + 0.10f * (1.0f - speed_norm);
    w[DIRECTOR_INTENT_RECOVER] = 0.16f + 0.26f * clampf_local(speed_norm - engage, 0.0f, 1.0f);

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
    state->director_time_remaining_sec = random_float_range(5.0f, 15.0f);
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
    const float low_engagement_boost = clampf_local(0.45f - state->engagement_score, 0.0f, 0.45f);
    const float trigger_prob_per_sec = (0.08f + low_engagement_boost) * director_tease_bias;
    if (random_float_range(0.0f, 1.0f) > trigger_prob_per_sec * clampf_local(dt_sec, 0.0f, 1.0f)) {
        return;
    }

    state->near_miss_phase = NEAR_MISS_BURST;
    const float catch_success = clampf_local(state->recent_catch_attempt_score, 0.0f, 1.0f);
    const int pass_bias = (int)floorf(0.5f + 3.0f * catch_success); // high success => longer tease runs.
    state->near_miss_passes_remaining = 2 + (rand() % 3) + pass_bias; // adaptive ~2-7 passes.
    state->near_miss_angle_rad = random_float_range(0.0f, 6.2831853f);
    state->near_miss_radius_scale = random_float_range(1.1f, 1.4f);
    state->near_miss_segment_time_sec = 0.0f;
    state->near_miss_segment_duration_sec = random_float_range(0.16f, 0.34f);
    state->near_miss_pause_time_sec = 0.0f;
    state->near_miss_direction = (rand() % 2 == 0) ? 1 : -1;
    (void)cat;
}

static int maybe_build_near_miss_tease_target(
    struct CatPlayState *state,
    const Yolov5CatTrackInfo &cat,
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

    const float radius = compute_catch_radius(cat, point_distance(cat_center, laser)) * state->near_miss_radius_scale;
    state->near_miss_angle_rad += (float)state->near_miss_direction * dt_sec * 11.0f;
    cv::Point2f tease_point(
        cat_center.x + cosf(state->near_miss_angle_rad) * radius,
        cat_center.y + sinf(state->near_miss_angle_rad) * radius);
    tease_point = clamp_point_to_frame(tease_point, frame_w, frame_h);

    state->near_miss_segment_time_sec += dt_sec;
    if (state->near_miss_segment_time_sec >= state->near_miss_segment_duration_sec) {
        state->near_miss_segment_time_sec = 0.0f;
        state->near_miss_segment_duration_sec = random_float_range(0.16f, 0.34f);
        state->near_miss_passes_remaining--;
        if ((rand() % 100) < 65) {
            state->near_miss_direction = -state->near_miss_direction;
        }
    }

    if (state->near_miss_passes_remaining <= 0) {
        state->near_miss_phase = NEAR_MISS_PAUSE;
        const float catch_success = clampf_local(state->recent_catch_attempt_score, 0.0f, 1.0f);
        const float pause_min = 0.30f + 0.35f * (1.0f - catch_success);
        const float pause_max = 0.65f + 0.55f * (1.0f - catch_success);
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
                                               const cv::Point2f &cat_center,
                                               const cv::Point2f &laser,
                                               float dt_sec) {
    state->oval_direction_cooldown_sec = clampf_local(state->oval_direction_cooldown_sec - dt_sec, 0.0f, 2.0f);

    // "Trying to catch" heuristic: laser gets very close to cat center.
    const float catch_radius = compute_catch_radius(cat, point_distance(cat_center, laser));
    if (point_distance(cat_center, laser) > catch_radius || state->oval_direction_cooldown_sec > 0.0f) {
        return;
    }

    // Randomized reaction on catch: each catch event gets its own flip probability,
    // then we sample whether orbit direction changes.
    const float flip_chance = random_float_range(0.45f, 0.85f);
    if (random_float_range(0.0f, 1.0f) < flip_chance) {
        state->oval_direction = -state->oval_direction;
    }

    // Debounce to avoid flipping every frame while the cat hovers near the laser.
    state->oval_direction_cooldown_sec = random_float_range(0.4f, 1.2f);
}

void init_cat_play_state(struct CatPlayState *state) {
    state->algorithm = CAT_PLAY_OVAL;
    state->last_cat_center = cv::Point2f(0.0f, 0.0f);
    state->cat_still_time_sec = 0.0f;
    state->velocity_initialized = 0;
    state->prev_velocity_cat_center = cv::Point2f(0.0f, 0.0f);
    state->cat_speed_px_per_sec_ema = 0.0f;
    state->was_within_catch_radius = 0;
    state->recent_catch_attempt_score = 0.0f;
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
    state->director_time_remaining_sec = random_float_range(5.0f, 15.0f);
    state->oval_phase = 0.0f;
    state->oval_direction = 1;
    state->oval_direction_cooldown_sec = 0.0f;
    state->near_miss_phase = NEAR_MISS_OFF;
    state->near_miss_passes_remaining = 0;
    state->near_miss_angle_rad = 0.0f;
    state->near_miss_radius_scale = 1.2f;
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
    const cv::Point2f &laser,
    int frame_index,
    int frame_w,
    int frame_h,
    float dt_sec,
    const char **algo_name_out) {
    (void)frame_index;
    const cv::Point2f cat_center(cat.x + cat.width * 0.5f, cat.y + cat.height * 0.5f);

    update_engagement_score(state, cat_center, laser, cat);
    update_cat_velocity_signal(state, cat_center, dt_sec);
    apply_novelty_decay_recovery(state, dt_sec);
    update_director_layer(state, dt_sec);

    // Anti-fatigue: every ~120s of active play, insert a short calm interval.
    state->session_time_sec += dt_sec;
    if (state->session_time_sec >= 120.0f && state->calm_time_sec <= 0.0f) {
        state->calm_time_sec = random_float_range(5.0f, 12.0f);
        state->session_time_sec = 0.0f;
    }
    if (state->calm_time_sec > 0.0f) {
        *algo_name_out = "calm_pause";
        state->calm_time_sec -= dt_sec;
        return laser;
    }

    if (state->hesitation_cooldown_sec > 0.0f) {
        state->hesitation_cooldown_sec = clampf_local(state->hesitation_cooldown_sec - dt_sec, 0.0f, 10.0f);
    }

    if (state->hesitation_pause_time_sec > 0.0f) {
        *algo_name_out = "hesitation_pause";
        state->hesitation_pause_time_sec -= dt_sec;
        return laser;
    }

    const float catch_radius = compute_catch_radius(cat, point_distance(cat_center, laser));
    const float speed_norm_for_hesitation = clampf_local(state->cat_speed_px_per_sec_ema / 220.0f, 0.0f, 1.0f);
    const float chase_dist = point_distance(cat_center, laser);
    const int high_speed_close_chase = (speed_norm_for_hesitation > 0.65f && chase_dist < (1.15f * catch_radius));

    if (high_speed_close_chase) {
        state->close_chase_time_sec += dt_sec;
    } else {
        state->close_chase_time_sec = clampf_local(state->close_chase_time_sec - dt_sec * 0.6f, 0.0f, 5.0f);
    }

    if (state->hesitation_cooldown_sec <= 0.0f && state->close_chase_time_sec >= 0.35f) {
        state->hesitation_pause_time_sec = random_float_range(0.2f, 0.8f);
        state->hesitation_cooldown_sec = random_float_range(1.4f, 2.4f);
        state->close_chase_time_sec = 0.0f;
        *algo_name_out = "hesitation_pause";
        return laser;
    }

    if (point_distance(cat_center, state->last_cat_center) < 10.0f) state->cat_still_time_sec += dt_sec;
    else {
        state->cat_still_time_sec = 0.0f;
        state->last_cat_center = cat_center;
    }
    if (state->cat_still_time_sec >= 60.0f) {
        state->algorithm = pick_engagement_ranked_alternate_algorithm(state);
        state->cat_still_time_sec = 0.0f;
    }

    float director_tease_bias = 1.0f;
    float oval_speed_scale = 1.0f;
    float zigzag_speed_scale = 1.0f;
    float zigzag_duration_scale = 1.0f;
    if (state->director_intent == DIRECTOR_INTENT_TEASE) {
        director_tease_bias = 1.7f;
        if (state->algorithm != CAT_PLAY_OVAL && random_float_range(0.0f, 1.0f) < 0.15f) {
            state->algorithm = CAT_PLAY_OVAL;
        }
    } else if (state->director_intent == DIRECTOR_INTENT_CHASE) {
        director_tease_bias = 0.8f;
        zigzag_speed_scale = 1.30f;
        zigzag_duration_scale = 0.75f;
        if (state->algorithm == CAT_PLAY_OVAL && random_float_range(0.0f, 1.0f) < 0.2f) {
            state->algorithm = CAT_PLAY_ZIGZAG_RETREAT;
            state->zigzag_phase = ZIGZAG_APPROACH;
        }
    } else if (state->director_intent == DIRECTOR_INTENT_POUNCE_WINDOW) {
        director_tease_bias = 0.7f;
        oval_speed_scale = 0.55f;
        if (state->algorithm != CAT_PLAY_OVAL && random_float_range(0.0f, 1.0f) < 0.25f) {
            state->algorithm = CAT_PLAY_OVAL;
        }
    } else { // DIRECTOR_INTENT_RECOVER
        director_tease_bias = 0.55f;
        oval_speed_scale = 0.5f;
        zigzag_speed_scale = 0.8f;
        zigzag_duration_scale = 1.2f;
    }

    if (state->algorithm == CAT_PLAY_OVAL) {
        maybe_flip_oval_direction_on_catch(state, cat, cat_center, laser, dt_sec);
        maybe_start_near_miss_tease(state, cat, dt_sec, director_tease_bias);

        cv::Point2f tease_target(0.0f, 0.0f);
        if (maybe_build_near_miss_tease_target(state, cat, cat_center, laser, frame_w, frame_h, dt_sec, algo_name_out, &tease_target)) {
            return tease_target;
        }

        *algo_name_out = "oval";
        const float speed_norm = clampf_local(state->cat_speed_px_per_sec_ema / 220.0f, 0.0f, 1.0f);
        const float direction = (state->oval_direction >= 0) ? 1.0f : -1.0f;
        const float oval_phase_step = (0.10f + 0.16f * speed_norm) * oval_speed_scale;
        state->oval_phase += direction * oval_phase_step;

        // If cat movement is low, occasionally bait with a dart pattern.
        if (speed_norm < 0.25f && random_float_range(0.0f, 1.0f) < (0.02f * clampf_local(dt_sec * 30.0f, 0.0f, 2.0f))) {
            state->algorithm = CAT_PLAY_STARE_DART;
            state->stare_dart_phase = STARE_DART_HOLD;
            state->stare_dart_hold_time_sec = random_float_range(2.0f, 6.0f);
            state->stare_dart_hold_point = laser;
        }
        return build_oval_target(cat, state->oval_phase, frame_w, frame_h);
    }

    if (state->algorithm == CAT_PLAY_STARE_DART) {
        *algo_name_out = "stare_dart";
        if (state->stare_dart_phase == STARE_DART_HOLD) {
            if (state->stare_dart_hold_time_sec <= 0.0f) {
                state->stare_dart_hold_time_sec = random_float_range(5.0f, 30.0f);
                state->stare_dart_hold_point = laser;
            }
            state->stare_dart_hold_time_sec -= dt_sec;
            if (state->stare_dart_hold_time_sec <= 0.0f) {
                state->stare_dart_phase = STARE_DART_DART;
                state->stare_dart_dart_target = random_point_outside_cat(cat, frame_w, frame_h);
                maybe_transition_with_probability(state, 10);
            }
            return clamp_point_to_frame(state->stare_dart_hold_point, frame_w, frame_h);
        }

        if (point_in_cat_bbox(state->stare_dart_dart_target, cat)) {
            state->stare_dart_dart_target = random_point_outside_cat(cat, frame_w, frame_h);
        }
        if (point_distance(laser, state->stare_dart_dart_target) < 20.0f) {
            state->stare_dart_phase = STARE_DART_HOLD;
            state->stare_dart_hold_time_sec = random_float_range(5.0f, 30.0f);
            state->stare_dart_hold_point = laser;
        }
        return clamp_point_to_frame(state->stare_dart_dart_target, frame_w, frame_h);
    }

    *algo_name_out = "zigzag_retreat";
    if (state->zigzag_phase == ZIGZAG_APPROACH) {
        state->zigzag_front_point = clamp_point_to_frame(cv::Point2f(cat_center.x, cat.y - cat.height * 0.25f), frame_w, frame_h);
        if (point_distance(laser, state->zigzag_front_point) < 18.0f) {
            state->zigzag_phase = ZIGZAG_SHAKE;
            state->zigzag_phase_time_sec = 0.0f;
        }
        return state->zigzag_front_point;
    }

    if (state->zigzag_phase == ZIGZAG_SHAKE) {
        state->zigzag_phase_time_sec += dt_sec;
        const float amp_x = clampf_local(cat.width * 0.45f, 12.0f, 80.0f);
        const float amp_y = clampf_local(cat.height * 0.18f, 6.0f, 35.0f);
        const float speed_norm = clampf_local(state->cat_speed_px_per_sec_ema / 220.0f, 0.0f, 1.0f);
        const float t = state->zigzag_phase_time_sec * (6.0f + 6.0f * speed_norm) * zigzag_speed_scale;
        const float saw = (fmodf(t, 2.0f) < 1.0f) ? 1.0f : -1.0f;
        cv::Point2f p(state->zigzag_front_point.x + saw * amp_x,
                      state->zigzag_front_point.y + sinf(t * 1.7f) * amp_y);
        const float shake_duration_sec = (3.2f - 1.6f * speed_norm) * zigzag_duration_scale;
        if (state->zigzag_phase_time_sec >= shake_duration_sec) {
            state->zigzag_phase = ZIGZAG_RETREAT;
            state->zigzag_retreat_point = furthest_frame_corner_from_point(laser, frame_w, frame_h);
        }
        return clamp_point_to_frame(p, frame_w, frame_h);
    }

    if (state->zigzag_phase == ZIGZAG_RETREAT) {
        state->zigzag_retreat_point = build_retreat_target_avoiding_cat(laser, cat, frame_w, frame_h);
        if (point_distance(laser, state->zigzag_retreat_point) < 20.0f) {
            state->zigzag_phase = ZIGZAG_RETURN;
            maybe_transition_with_probability(state, 10);
        }
        return state->zigzag_retreat_point;
    }

    state->zigzag_front_point = clamp_point_to_frame(cv::Point2f(cat_center.x, cat.y - cat.height * 0.25f), frame_w, frame_h);
    if (point_distance(laser, state->zigzag_front_point) < 18.0f) {
        state->zigzag_phase = ZIGZAG_SHAKE;
        state->zigzag_phase_time_sec = 0.0f;
    }
    return state->zigzag_front_point;
}
