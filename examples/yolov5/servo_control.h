#ifndef YOLOV5_SERVO_CONTROL_H_
#define YOLOV5_SERVO_CONTROL_H_

#include <stdbool.h>
#include <opencv2/core.hpp>
#if __has_include(<periphery/pwm.h>)
#include <periphery/pwm.h>
#include <periphery/gpio.h>
#else
#include <pwm.h>
#include <gpio.h>
#endif

struct ServoState {
    float pan_deg;
    float tilt_deg;
};

struct ServoPwm {
    pwm_t *handle;
    unsigned int chip;
    unsigned int channel;
};

struct MosfetPowerGpio {
    gpio_t *handle;
    const char *chip_path;
    unsigned int line;
};

int servo_pwm_open(struct ServoPwm *servo_pwm, unsigned int chip, unsigned int channel);
int servo_pwm_set_angle(struct ServoPwm *servo_pwm, float angle_deg);
int servo_pwm_enable(struct ServoPwm *servo_pwm);
void servo_pwm_close(struct ServoPwm *servo_pwm);

int mosfet_gpio_open(struct MosfetPowerGpio *mosfet_gpio, const char *chip_path, unsigned int line);
int mosfet_gpio_set(struct MosfetPowerGpio *mosfet_gpio, bool enabled);
void mosfet_gpio_close(struct MosfetPowerGpio *mosfet_gpio);

void update_servo_state(
    ServoState *servo_state,
    const cv::Point2f &current_laser,
    const cv::Point2f &target,
    int frame_width,
    int frame_height);

#endif
