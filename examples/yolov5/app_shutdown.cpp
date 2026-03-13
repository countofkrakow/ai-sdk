#include "app_shutdown.h"

#include <opencv2/highgui.hpp>

void app_runtime_shutdown(AppRuntime *rt, int laser_off_on_exit) {
    if (rt == NULL) return;

    if (laser_off_on_exit && !rt->cfg.dry_run) {
        mosfet_gpio_set(&rt->laser_gpio, false);
    }

    if (rt->context != NULL) {
        awnn_destroy(rt->context);
        rt->context = NULL;
    }
    awnn_uninit();

    if (!rt->cfg.dry_run) {
        mosfet_gpio_close(&rt->pan_power_gpio);
        mosfet_gpio_close(&rt->tilt_power_gpio);
        mosfet_gpio_close(&rt->laser_gpio);
        servo_pwm_close(&rt->pan_pwm);
        servo_pwm_close(&rt->tilt_pwm);
    }

    if (rt->camera.isOpened()) {
        rt->camera.release();
    }

    play_engine_destroy(rt->play_engine);
    rt->play_engine = NULL;

    pthread_mutex_destroy(&rt->frame_mailbox.mutex);
    pthread_cond_destroy(&rt->frame_mailbox.cond);
    pthread_mutex_destroy(&rt->inference_mailbox.mutex);

    debug_trace_close(&rt->trace);
    cv::destroyAllWindows();
}
