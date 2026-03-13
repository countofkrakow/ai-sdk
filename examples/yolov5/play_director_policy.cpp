#include "play_director_policy.h"
#include "play_algorithms_internal.h"

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
    if (total <= g_tuning.behavior.transition_weight_epsilon) {
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

void update_director_layer_policy(struct CatPlayState *state, float dt_sec) {
    state->director_time_remaining_sec -= dt_sec;
    if (state->director_time_remaining_sec > 0.0f) {
        return;
    }

    state->director_intent = choose_next_director_intent(state);
    state->director_time_remaining_sec =
        random_float_range(g_tuning.director.director_duration_min_sec, g_tuning.director.director_duration_max_sec);
}
