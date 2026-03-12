#ifndef YOLOV5_APP_CONFIG_H_
#define YOLOV5_APP_CONFIG_H_

#include <string>

struct AppConfig {
    const char *nbg_path;
    std::string camera_device;
    std::string replay_frames_dir;
    bool dry_run;

    unsigned int laser_brightness_percent;
    unsigned int laser_pwm_cycle_ticks;

    unsigned int pan_pwm_chip;
    unsigned int pan_pwm_channel;
    unsigned int tilt_pwm_chip;
    unsigned int tilt_pwm_channel;

    const char *mosfet_gpiochip_path;
    unsigned int pan_power_gpio_line;
    unsigned int tilt_power_gpio_line;
    unsigned int laser_gpio_line;

    int input_channels;
    float control_dt_sec;

    const char *play_tuning_json_path;
    const char *tracker_tuning_json_path;
    const char *inference_frame_file;
};

int app_parse_config(int argc, char **argv, AppConfig *cfg);
void app_print_usage(const char *argv0);

#endif
