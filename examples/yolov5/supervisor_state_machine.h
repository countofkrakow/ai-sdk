#ifndef SUPERVISOR_STATE_MACHINE_H
#define SUPERVISOR_STATE_MACHINE_H

#include <stdint.h>

enum SupervisorState {
    SUPERVISOR_STATE_IDLE = 0,
    SUPERVISOR_STATE_DETECT = 1,
    SUPERVISOR_STATE_WAKE = 2,
    SUPERVISOR_STATE_PLAY = 3,
    SUPERVISOR_STATE_DISENGAGE_TIMEOUT = 4,
    SUPERVISOR_STATE_SLEEP = 5,
};

enum WakeSource {
    WAKE_SOURCE_NONE = 0,
    WAKE_SOURCE_CAT = 1,
    WAKE_SOURCE_HUMAN_WAVE = 2,
    WAKE_SOURCE_HUMAN_PRESENCE = 3,
};

enum SessionMode {
    SESSION_MODE_NONE = 0,
    SESSION_MODE_ARMED = 1,
    SESSION_MODE_ACTIVE = 2,
    SESSION_MODE_LURE = 3,
    SESSION_MODE_COOLDOWN = 4,
};

enum CadenceMode {
    CADENCE_MODE_LOW = 0,
    CADENCE_MODE_MEDIUM = 1,
    CADENCE_MODE_HIGH = 2,
};

struct BinaryWindow {
    uint32_t bits;
    uint32_t valid_count;
    uint32_t window_size;
};

struct SupervisorConfig {
    int human_only_wake_enabled;
    uint64_t camera_stall_timeout_ms;
    uint64_t camera_recovery_confirm_ms;
    uint64_t detect_validation_min_ms;
    uint64_t detect_validation_timeout_ms;
    uint64_t wake_dwell_timeout_ms;
    uint64_t absence_to_cooldown_ms;
    uint64_t cooldown_to_sleep_ms;
    uint64_t minimum_play_dwell_ms;
    uint64_t lure_retry_window_ms;

    uint32_t cat_present_enter_hits;
    uint32_t cat_present_enter_window;
    uint32_t cat_present_exit_hits;
    uint32_t cat_present_exit_window;

    uint32_t human_present_enter_hits;
    uint32_t human_present_enter_window;
    uint32_t human_present_exit_hits;
    uint32_t human_present_exit_window;

    uint32_t wave_enter_hits;
    uint32_t wave_enter_window;

    uint32_t interest_low_hits;
    uint32_t interest_low_window;
};

struct DetectionSample {
    uint64_t now_ms;
    int camera_frame_valid;
    int cat_detected;
    int human_detected;
    int wave_detected;
    int cat_interest_low;
};

struct SupervisorOutputs {
    SupervisorState state;
    SessionMode session_mode;
    CadenceMode cadence_mode;
    WakeSource wake_source;

    int laser_enabled;
    int narration_enabled;
    int request_low_power;
    int request_active_power;
    int camera_deadman_active;
    int room_empty;
    int occupied;
    int transition_occurred;
};

struct SupervisorContext {
    SupervisorConfig cfg;

    SupervisorState state;
    SessionMode session_mode;
    WakeSource wake_source;
    CadenceMode cadence_mode;

    BinaryWindow cat_window;
    BinaryWindow human_window;
    BinaryWindow wave_window;
    BinaryWindow interest_low_window;

    int cat_present;
    int human_present;
    int wave_present;
    int cat_interest_low;
    int room_empty;
    int camera_deadman_active;

    uint64_t state_entered_ms;
    uint64_t last_valid_frame_ms;
    uint64_t camera_recovered_candidate_ms;
    uint64_t last_cat_or_human_seen_ms;
    uint64_t last_cat_seen_ms;
    uint64_t last_human_seen_ms;
    uint64_t last_cat_interest_ms;
};

SupervisorConfig supervisor_default_config(void);
void supervisor_init(SupervisorContext *ctx, const SupervisorConfig *cfg, uint64_t now_ms);
SupervisorOutputs supervisor_step(SupervisorContext *ctx, const DetectionSample *sample);

#endif
