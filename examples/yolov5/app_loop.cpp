#include "app_loop.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "image_utils.h"
#include "play_algorithms.h"
#include "servo_control.h"
#include "tracking_utils.h"
#include "yolov5_pre_process.h"

struct InferenceThreadArgs {
    Awnn_Context_t *context;
    const char *frame_file;
    FrameMailbox *frame_mailbox;
    InferenceMailbox *inference_mailbox;
    DebugTrace *trace;
};

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static float random_float_range(float min_v, float max_v) {
    const float r = (float)rand() / (float)RAND_MAX;
    return min_v + (max_v - min_v) * r;
}

static void apply_confidence_aware_servo_smoothing(ServoState *servo_state, const ServoState *pre_update_state, float confidence) {
    const float conf_norm = clampf((confidence - 0.25f) / 0.65f, 0.0f, 1.0f);
    const float alpha = 0.22f + 0.68f * conf_norm;
    servo_state->pan_deg = pre_update_state->pan_deg + alpha * (servo_state->pan_deg - pre_update_state->pan_deg);
    servo_state->tilt_deg = pre_update_state->tilt_deg + alpha * (servo_state->tilt_deg - pre_update_state->tilt_deg);
}

static float compute_laser_intensity_scale(enum PlayDirectorIntent director_intent, float engagement_score, const char *algo_name, float session_time_sec) {
    float intensity = 0.25f + 0.75f * clampf(engagement_score, 0.0f, 1.0f);
    if (director_intent == DIRECTOR_INTENT_CHASE) intensity += 0.20f;
    else if (director_intent == DIRECTOR_INTENT_TEASE) intensity -= 0.08f;
    else if (director_intent == DIRECTOR_INTENT_RECOVER) intensity -= 0.12f;

    if (algo_name != NULL &&
        (strcmp(algo_name, "hesitation_pause") == 0 || strcmp(algo_name, "near_miss_tease_pause") == 0)) {
        intensity *= 0.60f;
    }

    if (director_intent == DIRECTOR_INTENT_POUNCE_WINDOW) {
        const float phase = 2.0f * 3.1415926f * 1.8f * session_time_sec;
        const float pulse = sinf(phase);
        const float heartbeat = 0.88f + 0.12f * pulse * pulse;
        intensity *= heartbeat;
    }

    if (director_intent == DIRECTOR_INTENT_CHASE && intensity > 0.78f) {
        intensity += random_float_range(-0.05f, 0.05f);
    }

    return clampf(intensity, 0.08f, 1.0f);
}

static void apply_intensity_motion_style(ServoState *servo_state,
                                         const ServoState *pre_update_state,
                                         float intensity_scale,
                                         enum PlayDirectorIntent director_intent) {
    float style_alpha = 0.30f + 0.70f * clampf(intensity_scale, 0.0f, 1.0f);
    if (director_intent == DIRECTOR_INTENT_CHASE) {
        style_alpha = clampf(style_alpha + 0.08f, 0.0f, 1.0f);
    }
    if (director_intent == DIRECTOR_INTENT_TEASE || director_intent == DIRECTOR_INTENT_RECOVER) {
        style_alpha = clampf(style_alpha - 0.12f, 0.0f, 1.0f);
    }

    servo_state->pan_deg = pre_update_state->pan_deg + style_alpha * (servo_state->pan_deg - pre_update_state->pan_deg);
    servo_state->tilt_deg = pre_update_state->tilt_deg + style_alpha * (servo_state->tilt_deg - pre_update_state->tilt_deg);
}

static void *inference_thread_main(void *arg) {
    InferenceThreadArgs *args = (InferenceThreadArgs *)arg;

    while (1) {
        cv::Mat frame;
        unsigned long frame_seq = 0;
        double frame_ts = 0.0;

        pthread_mutex_lock(&args->frame_mailbox->mutex);
        while (!args->frame_mailbox->has_new_frame && !args->frame_mailbox->stop) {
            pthread_cond_wait(&args->frame_mailbox->cond, &args->frame_mailbox->mutex);
        }
        if (args->frame_mailbox->stop) {
            pthread_mutex_unlock(&args->frame_mailbox->mutex);
            break;
        }
        frame = args->frame_mailbox->latest_frame.clone();
        frame_seq = args->frame_mailbox->frame_seq;
        frame_ts = args->frame_mailbox->timestamp_sec;
        args->frame_mailbox->has_new_frame = 0;
        args->frame_mailbox->inference_running = 1;
        pthread_mutex_unlock(&args->frame_mailbox->mutex);

        Yolov5CatTrackInfo track_info = {0, 0, 0, 0, 0, 0};
        Yolov5CatDetections detections = {0};

        if (!frame.empty() && cv::imwrite(args->frame_file, frame)) {
            unsigned int file_size = 0;
            unsigned char *input_tensor_bytes = yolov5_pre_process(args->frame_file, &file_size);
            if (input_tensor_bytes != NULL) {
                void *input_buffers[] = {input_tensor_bytes};
                awnn_set_input_buffers(args->context, input_buffers);
                awnn_run(args->context);
                float **results = awnn_get_output_buffers(args->context);
                yolov5_post_process(args->frame_file, results, &track_info, &detections);
                free(input_tensor_bytes);
            }
        }

        pthread_mutex_lock(&args->inference_mailbox->mutex);
        args->inference_mailbox->latest_track = track_info;
        args->inference_mailbox->latest_detections = detections;
        args->inference_mailbox->source_frame_seq = frame_seq;
        args->inference_mailbox->source_timestamp_sec = frame_ts;
        args->inference_mailbox->has_cat_info = 1;
        pthread_mutex_unlock(&args->inference_mailbox->mutex);

        pthread_mutex_lock(&args->frame_mailbox->mutex);
        args->frame_mailbox->inference_running = 0;
        pthread_mutex_unlock(&args->frame_mailbox->mutex);
    }

    return NULL;
}

static int read_next_frame(AppRuntime *rt, FrameInputs *in) {
    if (rt->replay.enabled) {
        if (rt->replay.index >= rt->replay.frame_paths.size()) {
            return -1;
        }
        in->frame_bgr = cv::imread(rt->replay.frame_paths[rt->replay.index++], cv::IMREAD_COLOR);
    } else {
        cv::Mat raw_frame;
        if (!rt->camera.read(raw_frame) || raw_frame.empty()) {
            return -1;
        }
        in->frame_bgr = raw_frame;
    }

    if (in->frame_bgr.channels() == 4) cv::cvtColor(in->frame_bgr, in->frame_bgr, cv::COLOR_BGRA2BGR);
    else if (in->frame_bgr.channels() == 1) cv::cvtColor(in->frame_bgr, in->frame_bgr, cv::COLOR_GRAY2BGR);
    if (in->frame_bgr.channels() != rt->cfg.input_channels) {
        return -1;
    }

    in->frame_seq = (unsigned long)rt->frame_index;
    in->timestamp_sec = now_sec();
    return 0;
}

static void publish_frame_for_inference(AppRuntime *rt, const FrameInputs *in) {
    pthread_mutex_lock(&rt->frame_mailbox.mutex);
    rt->frame_mailbox.latest_frame = in->frame_bgr.clone();
    rt->frame_mailbox.frame_seq = in->frame_seq;
    rt->frame_mailbox.timestamp_sec = in->timestamp_sec;
    rt->frame_mailbox.has_new_frame = 1;
    pthread_cond_signal(&rt->frame_mailbox.cond);
    pthread_mutex_unlock(&rt->frame_mailbox.mutex);
}

static void snapshot_perception(AppRuntime *rt, PerceptionState *p) {
    memset(p, 0, sizeof(*p));

    pthread_mutex_lock(&rt->inference_mailbox.mutex);
    p->raw_track = rt->inference_mailbox.latest_track;
    p->raw_detections = rt->inference_mailbox.latest_detections;
    p->has_inference = rt->inference_mailbox.has_cat_info;
    p->inference_source_seq = rt->inference_mailbox.source_frame_seq;
    pthread_mutex_unlock(&rt->inference_mailbox.mutex);

    pthread_mutex_lock(&rt->frame_mailbox.mutex);
    p->inference_running = rt->frame_mailbox.inference_running;
    pthread_mutex_unlock(&rt->frame_mailbox.mutex);

    if (p->has_inference) {
        p->active_track = update_multi_cat_tracker_and_get_active(&rt->multi_cat_tracker, &p->raw_detections);
        if (!p->active_track.has_cat) {
            p->active_track = p->raw_track;
        }
    }
    p->filtered_track = filter_cat_track(&rt->track_filter, p->active_track.has_cat ? &p->active_track : NULL);
}

static void apply_actuation(AppRuntime *rt, const ActuationCommand *cmd) {
    if (!rt->cfg.dry_run) {
        servo_pwm_set_angle(&rt->pan_pwm, cmd->pan_deg);
        servo_pwm_set_angle(&rt->tilt_pwm, cmd->tilt_deg);
        mosfet_gpio_set(&rt->laser_gpio, cmd->laser_on != 0);
    }
}

int app_run_loop(AppRuntime *rt, volatile sig_atomic_t *stop_flag) {
    InferenceThreadArgs worker_args = {
        rt->context,
        rt->cfg.inference_frame_file,
        &rt->frame_mailbox,
        &rt->inference_mailbox,
        &rt->trace,
    };

    pthread_t inference_thread;
    if (pthread_create(&inference_thread, NULL, inference_thread_main, &worker_args) != 0) {
        return -1;
    }

    while (!(*stop_flag)) {
        FrameInputs in;
        if (read_next_frame(rt, &in) != 0) {
            if (rt->replay.enabled) {
                break;
            }
            usleep(100000);
            continue;
        }

        rt->last_frame_time = time(NULL);
        publish_frame_for_inference(rt, &in);

        PerceptionState perception;
        snapshot_perception(rt, &perception);

        ControlDecision decision = {};
        ActuationCommand cmd = {};

        if (perception.filtered_track.has_cat) {
            if (rt->frame_index == 0) {
                rt->virtual_laser_point = cv::Point2f((float)in.frame_bgr.cols * 0.5f, (float)in.frame_bgr.rows * 0.5f);
            }
            PlayEngineStepInput step_in = {};
            step_in.cat = perception.filtered_track;
            step_in.detection_confidence = perception.filtered_track.confidence;
            step_in.virtual_laser_point = rt->virtual_laser_point;
            step_in.frame_index = rt->frame_index;
            step_in.frame_w = in.frame_bgr.cols;
            step_in.frame_h = in.frame_bgr.rows;
            step_in.dt_sec = rt->cfg.control_dt_sec;
            PlayEngineStepOutput step_out = {};
            play_engine_step(rt->play_engine, &step_in, &step_out);

            decision.target_point = step_out.play_target;
            decision.algorithm_name = step_out.algo_name;
            decision.engagement_score = step_out.engagement_score;
            decision.director_intent = step_out.director_intent;
            decision.intensity_scale = compute_laser_intensity_scale(step_out.director_intent, step_out.engagement_score,
                                                                     step_out.algo_name, rt->play_session_time_sec);
            decision.has_target = 1;

            ServoState pre_update = rt->servo_state;
            cv::Point2f frame_center((float)in.frame_bgr.cols * 0.5f, (float)in.frame_bgr.rows * 0.5f);
            update_servo_state(&rt->servo_state, frame_center, decision.target_point, in.frame_bgr.cols, in.frame_bgr.rows);
            apply_confidence_aware_servo_smoothing(&rt->servo_state, &pre_update, perception.filtered_track.confidence);
            apply_intensity_motion_style(&rt->servo_state, &pre_update, decision.intensity_scale, decision.director_intent);

            const unsigned int effective_on_ticks = (unsigned int)(
                clampf((float)rt->laser_pwm_on_ticks * decision.intensity_scale, 0.0f, (float)rt->cfg.laser_pwm_cycle_ticks) + 0.5f);
            const int laser_on_this_tick = (effective_on_ticks > 0) && (rt->laser_pwm_tick < effective_on_ticks);
            rt->laser_pwm_tick = (rt->laser_pwm_tick + 1) % rt->cfg.laser_pwm_cycle_ticks;

            cmd.pan_deg = rt->servo_state.pan_deg;
            cmd.tilt_deg = rt->servo_state.tilt_deg;
            cmd.laser_on = laser_on_this_tick;
            cmd.effective_on_ticks = effective_on_ticks;

            rt->virtual_laser_point = decision.target_point;
            rt->play_session_time_sec += rt->cfg.control_dt_sec;

            debug_trace_log(&rt->trace, DEBUG_LOG_INFO, "PLAY",
                            "cat_conf=%.2f algo=%s engage=%.2f intensity=%.2f target=(%.1f,%.1f) servo=(%.2f,%.2f)",
                            perception.filtered_track.confidence,
                            decision.algorithm_name,
                            decision.engagement_score,
                            decision.intensity_scale,
                            decision.target_point.x,
                            decision.target_point.y,
                            cmd.pan_deg,
                            cmd.tilt_deg);
        } else {
            rt->servo_state.pan_deg = 0.0f;
            rt->servo_state.tilt_deg = 0.0f;
            cmd.pan_deg = rt->servo_state.pan_deg;
            cmd.tilt_deg = rt->servo_state.tilt_deg;
            cmd.laser_on = 1;
            cmd.effective_on_ticks = rt->laser_pwm_on_ticks;

            rt->virtual_laser_point = cv::Point2f((float)in.frame_bgr.cols * 0.5f, (float)in.frame_bgr.rows * 0.5f);
            rt->play_session_time_sec = 0.0f;
            play_engine_reset(rt->play_engine);

            debug_trace_log(&rt->trace, DEBUG_LOG_INFO, "TRACK", "No cat%s; holding center servo=(%.2f,%.2f)",
                            perception.inference_running ? " (inference busy)" : "",
                            cmd.pan_deg, cmd.tilt_deg);
        }

        apply_actuation(rt, &cmd);

        FrameTraceRow row = {};
        row.frame_seq = in.frame_seq;
        row.timestamp_sec = in.timestamp_sec;
        row.has_cat = perception.filtered_track.has_cat;
        row.confidence = perception.filtered_track.confidence;
        row.algorithm_name = decision.algorithm_name;
        row.engagement_score = decision.engagement_score;
        row.intensity_scale = decision.intensity_scale;
        row.target_x = decision.target_point.x;
        row.target_y = decision.target_point.y;
        row.pan_deg = cmd.pan_deg;
        row.tilt_deg = cmd.tilt_deg;
        row.laser_on = cmd.laser_on;
        debug_trace_write_row(&rt->trace, &row);

        cv::Mat detection = cv::imread("result.png");
        if (!detection.empty()) {
            cv::imshow("YOLOv5 Live Detection", detection);
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
        usleep(30000);
        rt->frame_index++;
    }

    pthread_mutex_lock(&rt->frame_mailbox.mutex);
    rt->frame_mailbox.stop = 1;
    pthread_cond_signal(&rt->frame_mailbox.cond);
    pthread_mutex_unlock(&rt->frame_mailbox.mutex);
    pthread_join(inference_thread, NULL);
    return 0;
}
