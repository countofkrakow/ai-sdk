#include "supervisor_state_machine.h"

#include <string.h>

static uint32_t mask_for_window(uint32_t window_size) {
    if (window_size >= 32U) {
        return 0xFFFFFFFFu;
    }
    return (1u << window_size) - 1u;
}

static void binary_window_reset(BinaryWindow *window, uint32_t window_size) {
    window->bits = 0u;
    window->valid_count = 0u;
    window->window_size = window_size;
}

static void binary_window_push(BinaryWindow *window, int value) {
    const uint32_t mask = mask_for_window(window->window_size);
    window->bits = ((window->bits << 1u) | (value ? 1u : 0u)) & mask;
    if (window->valid_count < window->window_size) {
        window->valid_count++;
    }
}

static uint32_t binary_window_popcount(const BinaryWindow *window) {
    uint32_t bits = window->bits;
    uint32_t count = 0u;
    while (bits != 0u) {
        count += bits & 1u;
        bits >>= 1u;
    }
    return count;
}

static int binary_window_hits(const BinaryWindow *window, uint32_t hits, uint32_t expected_window) {
    if (window->valid_count < expected_window) {
        return 0;
    }
    return binary_window_popcount(window) >= hits;
}

static int binary_window_absent_hits(const BinaryWindow *window, uint32_t absent_hits, uint32_t expected_window) {
    if (window->valid_count < expected_window) {
        return 0;
    }
    return (expected_window - binary_window_popcount(window)) >= absent_hits;
}

static void supervisor_transition(SupervisorContext *ctx,
                                  SupervisorOutputs *out,
                                  SupervisorState next_state,
                                  uint64_t now_ms) {
    if (ctx->state != next_state) {
        ctx->state = next_state;
        ctx->state_entered_ms = now_ms;
        out->transition_occurred = 1;
    }
}

static int room_is_occupied(const SupervisorContext *ctx) {
    return ctx->cat_present || ctx->human_present;
}

static int wake_signal_still_valid(const SupervisorContext *ctx) {
    switch (ctx->wake_source) {
        case WAKE_SOURCE_CAT:
            return ctx->cat_present;
        case WAKE_SOURCE_HUMAN_WAVE:
            return ctx->wave_present || ctx->human_present;
        case WAKE_SOURCE_HUMAN_PRESENCE:
            return ctx->human_present;
        case WAKE_SOURCE_NONE:
        default:
            return 0;
    }
}

static void update_windows(SupervisorContext *ctx, const DetectionSample *sample) {
    binary_window_push(&ctx->cat_window, sample->cat_detected);
    binary_window_push(&ctx->human_window, sample->human_detected);
    binary_window_push(&ctx->wave_window, sample->wave_detected);
    binary_window_push(&ctx->interest_low_window, sample->cat_interest_low);
}

static void update_presence_state(SupervisorContext *ctx, const DetectionSample *sample) {
    if (!ctx->cat_present) {
        if (binary_window_hits(&ctx->cat_window,
                               ctx->cfg.cat_present_enter_hits,
                               ctx->cfg.cat_present_enter_window)) {
            ctx->cat_present = 1;
        }
    } else if (binary_window_absent_hits(&ctx->cat_window,
                                         ctx->cfg.cat_present_exit_hits,
                                         ctx->cfg.cat_present_exit_window)) {
        ctx->cat_present = 0;
    }

    if (!ctx->human_present) {
        if (binary_window_hits(&ctx->human_window,
                               ctx->cfg.human_present_enter_hits,
                               ctx->cfg.human_present_enter_window)) {
            ctx->human_present = 1;
        }
    } else if (binary_window_absent_hits(&ctx->human_window,
                                         ctx->cfg.human_present_exit_hits,
                                         ctx->cfg.human_present_exit_window)) {
        ctx->human_present = 0;
    }

    ctx->wave_present = binary_window_hits(&ctx->wave_window,
                                           ctx->cfg.wave_enter_hits,
                                           ctx->cfg.wave_enter_window);
    ctx->cat_interest_low = binary_window_hits(&ctx->interest_low_window,
                                               ctx->cfg.interest_low_hits,
                                               ctx->cfg.interest_low_window);

    if (sample->cat_detected || sample->human_detected) {
        ctx->last_cat_or_human_seen_ms = sample->now_ms;
    }
    if (sample->cat_detected) {
        ctx->last_cat_seen_ms = sample->now_ms;
    }
    if (sample->human_detected) {
        ctx->last_human_seen_ms = sample->now_ms;
    }
    if (!sample->cat_interest_low && sample->cat_detected) {
        ctx->last_cat_interest_ms = sample->now_ms;
    }

    ctx->room_empty = !room_is_occupied(ctx);
}

static void update_camera_deadman(SupervisorContext *ctx, const DetectionSample *sample) {
    if (sample->camera_frame_valid) {
        ctx->last_valid_frame_ms = sample->now_ms;
        if (ctx->camera_deadman_active) {
            if (ctx->camera_recovered_candidate_ms == 0u) {
                ctx->camera_recovered_candidate_ms = sample->now_ms;
            } else if ((sample->now_ms - ctx->camera_recovered_candidate_ms) >= ctx->cfg.camera_recovery_confirm_ms) {
                ctx->camera_deadman_active = 0;
                ctx->camera_recovered_candidate_ms = 0u;
            }
        }
    } else {
        ctx->camera_recovered_candidate_ms = 0u;
        if ((sample->now_ms - ctx->last_valid_frame_ms) >= ctx->cfg.camera_stall_timeout_ms) {
            ctx->camera_deadman_active = 1;
        }
    }
}

SupervisorConfig supervisor_default_config(void) {
    SupervisorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.human_only_wake_enabled = 0;
    cfg.camera_stall_timeout_ms = 2000u;
    cfg.camera_recovery_confirm_ms = 500u;
    cfg.detect_validation_min_ms = 500u;
    cfg.detect_validation_timeout_ms = 3000u;
    cfg.wake_dwell_timeout_ms = 3000u;
    cfg.absence_to_cooldown_ms = 45000u;
    cfg.cooldown_to_sleep_ms = 15000u;
    cfg.minimum_play_dwell_ms = 5000u;
    cfg.lure_retry_window_ms = 15000u;

    cfg.cat_present_enter_hits = 3u;
    cfg.cat_present_enter_window = 5u;
    cfg.cat_present_exit_hits = 10u;
    cfg.cat_present_exit_window = 12u;

    cfg.human_present_enter_hits = 3u;
    cfg.human_present_enter_window = 5u;
    cfg.human_present_exit_hits = 10u;
    cfg.human_present_exit_window = 12u;

    cfg.wave_enter_hits = 2u;
    cfg.wave_enter_window = 4u;

    cfg.interest_low_hits = 3u;
    cfg.interest_low_window = 5u;
    return cfg;
}

void supervisor_init(SupervisorContext *ctx, const SupervisorConfig *cfg, uint64_t now_ms) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->cfg = cfg ? *cfg : supervisor_default_config();
    ctx->state = SUPERVISOR_STATE_IDLE;
    ctx->session_mode = SESSION_MODE_NONE;
    ctx->wake_source = WAKE_SOURCE_NONE;
    ctx->cadence_mode = CADENCE_MODE_LOW;
    ctx->state_entered_ms = now_ms;
    ctx->last_valid_frame_ms = now_ms;
    ctx->last_cat_or_human_seen_ms = now_ms;
    ctx->last_cat_seen_ms = now_ms;
    ctx->last_human_seen_ms = now_ms;
    ctx->last_cat_interest_ms = now_ms;

    binary_window_reset(&ctx->cat_window, ctx->cfg.cat_present_exit_window);
    binary_window_reset(&ctx->human_window, ctx->cfg.human_present_exit_window);
    binary_window_reset(&ctx->wave_window, ctx->cfg.wave_enter_window);
    binary_window_reset(&ctx->interest_low_window, ctx->cfg.interest_low_window);
}

SupervisorOutputs supervisor_step(SupervisorContext *ctx, const DetectionSample *sample) {
    SupervisorOutputs out;
    memset(&out, 0, sizeof(out));

    update_camera_deadman(ctx, sample);
    update_windows(ctx, sample);
    update_presence_state(ctx, sample);

    if (ctx->camera_deadman_active) {
        ctx->session_mode = SESSION_MODE_COOLDOWN;
        ctx->cadence_mode = CADENCE_MODE_LOW;
        ctx->wake_source = WAKE_SOURCE_NONE;
        supervisor_transition(ctx, &out, SUPERVISOR_STATE_IDLE, sample->now_ms);
    } else {
        switch (ctx->state) {
            case SUPERVISOR_STATE_IDLE:
            case SUPERVISOR_STATE_SLEEP:
                ctx->session_mode = SESSION_MODE_NONE;
                ctx->cadence_mode = CADENCE_MODE_LOW;
                if (ctx->cat_present) {
                    ctx->wake_source = WAKE_SOURCE_CAT;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_DETECT, sample->now_ms);
                } else if (ctx->wave_present) {
                    ctx->wake_source = WAKE_SOURCE_HUMAN_WAVE;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_DETECT, sample->now_ms);
                } else if (ctx->cfg.human_only_wake_enabled && ctx->human_present) {
                    ctx->wake_source = WAKE_SOURCE_HUMAN_PRESENCE;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_DETECT, sample->now_ms);
                }
                break;

            case SUPERVISOR_STATE_DETECT: {
                ctx->session_mode = SESSION_MODE_ARMED;
                ctx->cadence_mode = CADENCE_MODE_MEDIUM;
                const uint64_t detect_age_ms = sample->now_ms - ctx->state_entered_ms;
                const int signal_valid = wake_signal_still_valid(ctx);
                if (!signal_valid || ctx->room_empty) {
                    ctx->wake_source = WAKE_SOURCE_NONE;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_IDLE, sample->now_ms);
                } else if (detect_age_ms >= ctx->cfg.detect_validation_min_ms) {
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_WAKE, sample->now_ms);
                } else if (detect_age_ms >= ctx->cfg.detect_validation_timeout_ms) {
                    ctx->wake_source = WAKE_SOURCE_NONE;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_IDLE, sample->now_ms);
                }
                break;
            }

            case SUPERVISOR_STATE_WAKE: {
                ctx->session_mode = SESSION_MODE_ARMED;
                ctx->cadence_mode = CADENCE_MODE_HIGH;
                const uint64_t wake_age_ms = sample->now_ms - ctx->state_entered_ms;
                if (ctx->cat_present) {
                    ctx->session_mode = SESSION_MODE_ACTIVE;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_PLAY, sample->now_ms);
                } else if (((ctx->wake_source == WAKE_SOURCE_HUMAN_WAVE) ||
                            (ctx->wake_source == WAKE_SOURCE_HUMAN_PRESENCE)) &&
                           ctx->human_present) {
                    ctx->session_mode = SESSION_MODE_LURE;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_PLAY, sample->now_ms);
                } else if (!wake_signal_still_valid(ctx) || wake_age_ms >= ctx->cfg.wake_dwell_timeout_ms) {
                    ctx->session_mode = SESSION_MODE_COOLDOWN;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_DISENGAGE_TIMEOUT, sample->now_ms);
                }
                break;
            }

            case SUPERVISOR_STATE_PLAY: {
                ctx->cadence_mode = CADENCE_MODE_HIGH;
                const uint64_t since_presence_ms = sample->now_ms - ctx->last_cat_or_human_seen_ms;
                if (ctx->cat_present && !ctx->cat_interest_low) {
                    ctx->session_mode = SESSION_MODE_ACTIVE;
                } else if (ctx->human_present) {
                    ctx->session_mode = SESSION_MODE_LURE;
                } else {
                    ctx->session_mode = SESSION_MODE_COOLDOWN;
                }

                if (ctx->room_empty &&
                    since_presence_ms >= ctx->cfg.absence_to_cooldown_ms &&
                    (sample->now_ms - ctx->state_entered_ms) >= ctx->cfg.minimum_play_dwell_ms) {
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_DISENGAGE_TIMEOUT, sample->now_ms);
                }
                break;
            }

            case SUPERVISOR_STATE_DISENGAGE_TIMEOUT: {
                ctx->session_mode = SESSION_MODE_COOLDOWN;
                ctx->cadence_mode = CADENCE_MODE_MEDIUM;
                const uint64_t cooldown_age_ms = sample->now_ms - ctx->state_entered_ms;
                if (ctx->cat_present || ctx->wave_present ||
                    (ctx->cfg.human_only_wake_enabled && ctx->human_present)) {
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_WAKE, sample->now_ms);
                } else if (cooldown_age_ms >= ctx->cfg.cooldown_to_sleep_ms) {
                    ctx->wake_source = WAKE_SOURCE_NONE;
                    supervisor_transition(ctx, &out, SUPERVISOR_STATE_SLEEP, sample->now_ms);
                }
                break;
            }
        }
    }

    out.state = ctx->state;
    out.session_mode = ctx->session_mode;
    out.cadence_mode = ctx->cadence_mode;
    out.wake_source = ctx->wake_source;
    out.camera_deadman_active = ctx->camera_deadman_active;
    out.room_empty = ctx->room_empty;
    out.occupied = room_is_occupied(ctx);

    out.request_active_power = (ctx->state == SUPERVISOR_STATE_WAKE || ctx->state == SUPERVISOR_STATE_PLAY);
    out.request_low_power = (ctx->state == SUPERVISOR_STATE_IDLE ||
                             ctx->state == SUPERVISOR_STATE_SLEEP ||
                             ctx->camera_deadman_active);
    out.laser_enabled = (ctx->state == SUPERVISOR_STATE_PLAY &&
                         ctx->cat_present &&
                         !ctx->room_empty &&
                         !ctx->camera_deadman_active);
    out.narration_enabled = ((ctx->state == SUPERVISOR_STATE_WAKE || ctx->state == SUPERVISOR_STATE_PLAY) &&
                             !ctx->room_empty &&
                             !ctx->camera_deadman_active);
    return out;
}
