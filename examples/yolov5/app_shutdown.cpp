#include "app_shutdown.h"

#include <opencv2/highgui.hpp>

void app_runtime_shutdown(AppRuntime *rt, int laser_off_on_exit) {
    if (rt == NULL) return;
    debug_trace_log(&rt->trace, DEBUG_LOG_INFO, "SHUTDOWN",
                    "Runtime shutdown start: laser_off=%d dry_run=%d context=%p",
                    laser_off_on_exit,
                    rt->cfg.dry_run ? 1 : 0,
                    (void *)rt->context);

    if (laser_off_on_exit && !rt->cfg.dry_run) {
        mosfet_gpio_set(&rt->laser_gpio, false);
    }

    if (rt->context != NULL) {
        awnn_destroy(rt->context);
        rt->context = NULL;
    }
    if (rt->awnn_initialized) {
        awnn_uninit();
        rt->awnn_initialized = 0;
    }

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

    if (rt->frame_mailbox_mutex_initialized) {
        pthread_mutex_destroy(&rt->frame_mailbox.mutex);
        rt->frame_mailbox_mutex_initialized = 0;
    }
    if (rt->frame_mailbox_cond_initialized) {
        pthread_cond_destroy(&rt->frame_mailbox.cond);
        rt->frame_mailbox_cond_initialized = 0;
    }
    if (rt->inference_mailbox_mutex_initialized) {
        pthread_mutex_destroy(&rt->inference_mailbox.mutex);
        rt->inference_mailbox_mutex_initialized = 0;
    }

    debug_trace_close(&rt->trace);
    cv::destroyAllWindows();
}
