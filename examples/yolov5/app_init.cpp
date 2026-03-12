#include "app_init.h"

#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "play_algorithms.h"
#include "tracking_utils.h"

static void print_pwm_sysfs_overview(void) {
    DIR *dir = opendir("/sys/class/pwm");
    if (dir == NULL) {
        fprintf(stderr, "PWM debug: /sys/class/pwm is unavailable on this system.\n");
        return;
    }

    fprintf(stderr, "PWM debug: discovered pwmchips in /sys/class/pwm:\n");
    struct dirent *entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "pwmchip", 7) != 0) continue;
        char npwm_path[256];
        snprintf(npwm_path, sizeof(npwm_path), "/sys/class/pwm/%s/npwm", entry->d_name);
        FILE *f = fopen(npwm_path, "r");
        int npwm = -1;
        if (f != NULL) {
            if (fscanf(f, "%d", &npwm) != 1) npwm = -1;
            fclose(f);
        }
        fprintf(stderr, "  - %s (npwm=%d)\n", entry->d_name, npwm);
    }
    closedir(dir);
}

static void probe_servo_signs_of_life(struct ServoPwm *pan_pwm, struct ServoPwm *tilt_pwm, bool dry_run) {
    if (dry_run) return;
    const float pan_probe_points[] = {-20.0f, 20.0f, 0.0f};
    for (unsigned int i = 0; i < sizeof(pan_probe_points) / sizeof(pan_probe_points[0]); ++i) {
        servo_pwm_set_angle(pan_pwm, pan_probe_points[i]);
        servo_pwm_set_angle(tilt_pwm, 0.0f);
        usleep(220000);
    }

    const float tilt_probe_points[] = {-15.0f, 15.0f, 0.0f};
    for (unsigned int i = 0; i < sizeof(tilt_probe_points) / sizeof(tilt_probe_points[0]); ++i) {
        servo_pwm_set_angle(pan_pwm, 0.0f);
        servo_pwm_set_angle(tilt_pwm, tilt_probe_points[i]);
        usleep(220000);
    }
}

int app_runtime_init(const AppConfig *cfg, AppRuntime *rt) {
    if (cfg == NULL || rt == NULL) return -1;
    memset(rt, 0, sizeof(*rt));
    rt->cfg = *cfg;

    srand((unsigned int)time(NULL));
    debug_trace_init(&rt->trace, DEBUG_LOG_INFO, "examples/yolov5/frame_trace.csv");

    if (load_cat_play_tuning_json(cfg->play_tuning_json_path) == 0) {
        debug_trace_log(&rt->trace, DEBUG_LOG_INFO, "INIT", "Loaded play tuning: %s", cfg->play_tuning_json_path);
    } else {
        debug_trace_log(&rt->trace, DEBUG_LOG_WARN, "INIT", "Play tuning not loaded (%s); using defaults", cfg->play_tuning_json_path);
    }
    reset_tracker_tuning_defaults();
    if (load_tracker_tuning_json(cfg->tracker_tuning_json_path) == 0) {
        debug_trace_log(&rt->trace, DEBUG_LOG_INFO, "INIT", "Loaded tracker tuning: %s", cfg->tracker_tuning_json_path);
    } else {
        debug_trace_log(&rt->trace, DEBUG_LOG_WARN, "INIT", "Tracker tuning not loaded (%s); using defaults", cfg->tracker_tuning_json_path);
    }

    rt->laser_pwm_tick = 0;
    rt->laser_pwm_on_ticks = (cfg->laser_brightness_percent * cfg->laser_pwm_cycle_ticks) / 100;

    pthread_mutex_init(&rt->frame_mailbox.mutex, NULL);
    pthread_cond_init(&rt->frame_mailbox.cond, NULL);
    pthread_mutex_init(&rt->inference_mailbox.mutex, NULL);

    rt->context = NULL;
    if (!cfg->replay_frames_dir.empty()) {
        rt->replay.enabled = 1;
        cv::glob(cv::String(cfg->replay_frames_dir + "/*"), rt->replay.frame_paths, false);
        rt->replay.index = 0;
        if (rt->replay.frame_paths.empty()) {
            debug_trace_log(&rt->trace, DEBUG_LOG_ERROR, "INIT", "Replay requested but no frames found in %s", cfg->replay_frames_dir.c_str());
            return -1;
        }
        debug_trace_log(&rt->trace, DEBUG_LOG_INFO, "INIT", "Replay mode with %zu frames", rt->replay.frame_paths.size());
    } else {
        rt->camera.open(cfg->camera_device, cv::CAP_V4L2);
        if (!rt->camera.isOpened()) {
            debug_trace_log(&rt->trace, DEBUG_LOG_ERROR, "INIT", "Failed to open camera: %s", cfg->camera_device.c_str());
            return -1;
        }
    }

    awnn_init();
    rt->context = awnn_create(cfg->nbg_path);
    if (rt->context == NULL) {
        debug_trace_log(&rt->trace, DEBUG_LOG_ERROR, "INIT", "Failed to create NPU context: %s", cfg->nbg_path);
        return -1;
    }

    if (!cfg->dry_run) {
        if (mosfet_gpio_open(&rt->pan_power_gpio, cfg->mosfet_gpiochip_path, cfg->pan_power_gpio_line) < 0 ||
            mosfet_gpio_open(&rt->tilt_power_gpio, cfg->mosfet_gpiochip_path, cfg->tilt_power_gpio_line) < 0 ||
            mosfet_gpio_open(&rt->laser_gpio, cfg->mosfet_gpiochip_path, cfg->laser_gpio_line) < 0) {
            return -1;
        }

        if (mosfet_gpio_set(&rt->pan_power_gpio, true) < 0 ||
            mosfet_gpio_set(&rt->tilt_power_gpio, true) < 0 ||
            mosfet_gpio_set(&rt->laser_gpio, true) < 0) {
            return -1;
        }

        if (servo_pwm_open(&rt->pan_pwm, cfg->pan_pwm_chip, cfg->pan_pwm_channel) < 0 ||
            servo_pwm_open(&rt->tilt_pwm, cfg->tilt_pwm_chip, cfg->tilt_pwm_channel) < 0 ||
            servo_pwm_set_angle(&rt->pan_pwm, 0.0f) < 0 ||
            servo_pwm_set_angle(&rt->tilt_pwm, 0.0f) < 0 ||
            servo_pwm_enable(&rt->pan_pwm) < 0 ||
            servo_pwm_enable(&rt->tilt_pwm) < 0) {
            print_pwm_sysfs_overview();
            return -1;
        }

        probe_servo_signs_of_life(&rt->pan_pwm, &rt->tilt_pwm, cfg->dry_run);
    }

    init_multi_cat_tracker_state(&rt->multi_cat_tracker);
    rt->servo_state.pan_deg = 0.0f;
    rt->servo_state.tilt_deg = 0.0f;
    rt->virtual_laser_point = cv::Point2f(0.0f, 0.0f);
    rt->play_engine = play_engine_init();
    if (rt->play_engine == NULL) {
        return -1;
    }

    rt->frame_index = 0;
    rt->play_session_time_sec = 0.0f;
    rt->servo_rails_powered = cfg->dry_run ? 0 : 1;
    rt->deadman_active = 0;
    rt->last_frame_time = time(NULL);
    return 0;
}
