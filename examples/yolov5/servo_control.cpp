#include "servo_control.h"

#include <stdio.h>

static float clampf_local(float value, float min_v, float max_v) {
    return (value < min_v) ? min_v : ((value > max_v) ? max_v : value);
}

int servo_pwm_open(struct ServoPwm *servo_pwm, unsigned int chip, unsigned int channel) {
    servo_pwm->handle = pwm_new();
    if (servo_pwm->handle == NULL) {
        fprintf(stderr, "pwm_new failed for chip=%u channel=%u\n", chip, channel);
        return -1;
    }

    servo_pwm->chip = chip;
    servo_pwm->channel = channel;

    if (pwm_open(servo_pwm->handle, chip, channel) < 0) {
        fprintf(stderr, "pwm_open failed for chip=%u channel=%u\n", chip, channel);
        pwm_free(servo_pwm->handle);
        servo_pwm->handle = NULL;
        return -1;
    }

    if (pwm_set_period_ns(servo_pwm->handle, 20000000ULL) < 0) {
        fprintf(stderr, "pwm_set_period_ns failed for chip=%u channel=%u\n", chip, channel);
        pwm_close(servo_pwm->handle);
        pwm_free(servo_pwm->handle);
        servo_pwm->handle = NULL;
        return -1;
    }

    return 0;
}

int servo_pwm_set_angle(struct ServoPwm *servo_pwm, float angle_deg) {
    if (servo_pwm->handle == NULL) {
        return -1;
    }

    const float clamped = clampf_local(angle_deg, -45.0f, 45.0f);
    const float ratio = (clamped + 45.0f) / 90.0f;
    const unsigned long long pulse_ns = (unsigned long long)(1000000.0f + ratio * 1000000.0f);

    if (pwm_set_duty_cycle_ns(servo_pwm->handle, pulse_ns) < 0) {
        fprintf(stderr, "pwm_set_duty_cycle_ns failed for chip=%u channel=%u\n", servo_pwm->chip, servo_pwm->channel);
        return -1;
    }

    return 0;
}

int servo_pwm_enable(struct ServoPwm *servo_pwm) {
    if (servo_pwm->handle == NULL) {
        return -1;
    }

    if (pwm_enable(servo_pwm->handle) < 0) {
        fprintf(stderr, "pwm_enable failed for chip=%u channel=%u\n", servo_pwm->chip, servo_pwm->channel);
        return -1;
    }

    return 0;
}

void servo_pwm_close(struct ServoPwm *servo_pwm) {
    if (servo_pwm->handle == NULL) {
        return;
    }

    pwm_disable(servo_pwm->handle);
    pwm_close(servo_pwm->handle);
    pwm_free(servo_pwm->handle);
    servo_pwm->handle = NULL;
}

int mosfet_gpio_open(struct MosfetPowerGpio *mosfet_gpio, const char *chip_path, unsigned int line) {
    mosfet_gpio->handle = gpio_new();
    if (mosfet_gpio->handle == NULL) {
        fprintf(stderr, "gpio_new failed for %s line=%u\n", chip_path, line);
        return -1;
    }

    mosfet_gpio->chip_path = chip_path;
    mosfet_gpio->line = line;

    if (gpio_open(mosfet_gpio->handle, chip_path, line, GPIO_DIR_OUT_LOW) < 0) {
        fprintf(stderr, "gpio_open failed for %s line=%u\n", chip_path, line);
        gpio_free(mosfet_gpio->handle);
        mosfet_gpio->handle = NULL;
        return -1;
    }

    return 0;
}

int mosfet_gpio_set(struct MosfetPowerGpio *mosfet_gpio, bool enabled) {
    if (mosfet_gpio->handle == NULL) {
        return -1;
    }

    if (gpio_write(mosfet_gpio->handle, enabled) < 0) {
        fprintf(stderr, "gpio_write failed for %s line=%u\n", mosfet_gpio->chip_path, mosfet_gpio->line);
        return -1;
    }

    return 0;
}

void mosfet_gpio_close(struct MosfetPowerGpio *mosfet_gpio) {
    if (mosfet_gpio->handle == NULL) {
        return;
    }

    gpio_write(mosfet_gpio->handle, false);
    gpio_close(mosfet_gpio->handle);
    gpio_free(mosfet_gpio->handle);
    mosfet_gpio->handle = NULL;
}

void update_servo_state(
    ServoState *servo_state,
    const cv::Point2f &current_laser,
    const cv::Point2f &target,
    int frame_width,
    int frame_height) {
    const float pan_gain = 16.0f;
    const float tilt_gain = 16.0f;
    const float max_step_deg = 1.8f;

    float err_x = (target.x - current_laser.x) / (float)frame_width;
    float err_y = (target.y - current_laser.y) / (float)frame_height;

    float pan_delta = clampf_local(err_x * pan_gain, -max_step_deg, max_step_deg);
    float tilt_delta = clampf_local(err_y * tilt_gain, -max_step_deg, max_step_deg);

    servo_state->pan_deg = clampf_local(servo_state->pan_deg + pan_delta, -45.0f, 45.0f);
    servo_state->tilt_deg = clampf_local(servo_state->tilt_deg + tilt_delta, -45.0f, 45.0f);
}
