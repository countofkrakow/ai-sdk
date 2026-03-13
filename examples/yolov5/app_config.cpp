#include "app_config.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_brightness_percent(const char *arg, unsigned int *brightness_percent) {
    if (arg == NULL || brightness_percent == NULL) {
        return -1;
    }

    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0' || parsed > 100UL) {
        return -1;
    }

    *brightness_percent = (unsigned int)parsed;
    return 0;
}

void app_print_usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s <nbg> [camera_device] [laser_brightness_percent] [--dry-run] [--replay <frames_dir>]\n"
            "  nbg: path to YOLOv5 .nb model\n"
            "  camera_device: optional V4L2 node (default: /dev/video0)\n"
            "  laser_brightness_percent: optional integer 0..100 (default: 100)\n"
            "  --dry-run: run logic without writing servo/GPIO outputs\n"
            "  --replay <frames_dir>: replay image sequence instead of live camera\n",
            argv0);
}

int app_parse_config(int argc, char **argv, AppConfig *cfg) {
    if (cfg == NULL || argc < 2) {
        return -1;
    }

    cfg->nbg_path = argv[1];
    cfg->camera_device = "/dev/video0";
    cfg->replay_frames_dir.clear();
    cfg->dry_run = false;

    cfg->laser_brightness_percent = 100;
    cfg->laser_pwm_cycle_ticks = 20;

    cfg->pan_pwm_chip = 10;
    cfg->pan_pwm_channel = 1;
    cfg->tilt_pwm_chip = 10;
    cfg->tilt_pwm_channel = 2;

    cfg->mosfet_gpiochip_path = "/dev/gpiochip0";
    cfg->pan_power_gpio_line = 32;
    cfg->tilt_power_gpio_line = 33;
    cfg->laser_gpio_line = 35;

    cfg->input_channels = 3;
    cfg->control_dt_sec = 0.03f;

    cfg->play_tuning_json_path = "examples/yolov5/play_tuning.json";
    cfg->tracker_tuning_json_path = "examples/yolov5/tracker_tuning.json";
    cfg->inference_frame_file = "live_frame.jpg";

    int positional_seen = 0;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--dry-run") == 0) {
            cfg->dry_run = true;
            continue;
        }
        if (strcmp(argv[i], "--replay") == 0) {
            if (i + 1 >= argc) {
                return -1;
            }
            cfg->replay_frames_dir = argv[++i];
            continue;
        }

        if (positional_seen == 0) {
            if (parse_brightness_percent(argv[i], &cfg->laser_brightness_percent) != 0) {
                cfg->camera_device = argv[i];
            }
            positional_seen++;
            continue;
        }

        if (positional_seen == 1) {
            if (parse_brightness_percent(argv[i], &cfg->laser_brightness_percent) != 0) {
                return -1;
            }
            positional_seen++;
            continue;
        }

        return -1;
    }

    return 0;
}
