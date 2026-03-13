#include "tracking_utils.h"

#include <math.h>
#include <stdio.h>
#include <vector>
#include <opencv2/imgproc.hpp>

struct TrackerTuningConfig {
    float selection_confidence_weight;
    float selection_age_weight;
    float selection_stability_weight;
    float selection_continuity_weight;
    float selection_engagement_weight;
    float age_norm_frames;
    float stability_norm_frames;
    float missed_norm_frames;
    float continuity_distance_norm_px;

    float match_iou_threshold;
    int stale_track_frames;

    float engagement_decay_keep;
    float response_diag_scale;
    float response_signal_motion_weight;
    float response_signal_confidence_weight;
    float response_ema_keep;
    float response_ema_gain;

    int filter_hold_miss_frames;
    int identity_lock_init_frames;
    float identity_jump_dist_px;
    float identity_confidence_override;
    float smoothing_alpha_high;
    float smoothing_alpha_low;
    float smoothing_high_confidence_threshold;
};

static TrackerTuningConfig g_tracker_tuning;
static int g_tracker_tuning_initialized = 0;

static void maybe_read_float(const cv::FileNode &n, float *dst) {
    if (n.empty() || dst == NULL) {
        return;
    }
    *dst = (float)n;
}

static void maybe_read_int(const cv::FileNode &n, int *dst) {
    if (n.empty() || dst == NULL) {
        return;
    }
    *dst = (int)n;
}

void reset_tracker_tuning_defaults(void) {
    g_tracker_tuning.selection_confidence_weight = 0.38f;
    g_tracker_tuning.selection_age_weight = 0.16f;
    g_tracker_tuning.selection_stability_weight = 0.18f;
    g_tracker_tuning.selection_continuity_weight = 0.10f;
    g_tracker_tuning.selection_engagement_weight = 0.18f;
    g_tracker_tuning.age_norm_frames = 24.0f;
    g_tracker_tuning.stability_norm_frames = 12.0f;
    g_tracker_tuning.missed_norm_frames = 12.0f;
    g_tracker_tuning.continuity_distance_norm_px = 220.0f;

    g_tracker_tuning.match_iou_threshold = 0.2f;
    g_tracker_tuning.stale_track_frames = 12;

    g_tracker_tuning.engagement_decay_keep = 0.97f;
    g_tracker_tuning.response_diag_scale = 0.55f;
    g_tracker_tuning.response_signal_motion_weight = 0.65f;
    g_tracker_tuning.response_signal_confidence_weight = 0.35f;
    g_tracker_tuning.response_ema_keep = 0.85f;
    g_tracker_tuning.response_ema_gain = 0.15f;

    g_tracker_tuning.filter_hold_miss_frames = 12;
    g_tracker_tuning.identity_lock_init_frames = 10;
    g_tracker_tuning.identity_jump_dist_px = 120.0f;
    g_tracker_tuning.identity_confidence_override = 0.78f;
    g_tracker_tuning.smoothing_alpha_high = 0.35f;
    g_tracker_tuning.smoothing_alpha_low = 0.2f;
    g_tracker_tuning.smoothing_high_confidence_threshold = 0.7f;

    g_tracker_tuning_initialized = 1;
}


static void validate_tracker_tuning() {
    if (g_tracker_tuning.age_norm_frames <= 0.01f) g_tracker_tuning.age_norm_frames = 24.0f;
    if (g_tracker_tuning.stability_norm_frames <= 0.01f) g_tracker_tuning.stability_norm_frames = 12.0f;
    if (g_tracker_tuning.missed_norm_frames <= 0.01f) g_tracker_tuning.missed_norm_frames = 12.0f;
    if (g_tracker_tuning.continuity_distance_norm_px <= 1.0f) g_tracker_tuning.continuity_distance_norm_px = 220.0f;
    if (g_tracker_tuning.match_iou_threshold < 0.0f) g_tracker_tuning.match_iou_threshold = 0.0f;
    if (g_tracker_tuning.match_iou_threshold > 1.0f) g_tracker_tuning.match_iou_threshold = 1.0f;
    if (g_tracker_tuning.stale_track_frames < 1) g_tracker_tuning.stale_track_frames = 1;
    if (g_tracker_tuning.engagement_decay_keep < 0.0f) g_tracker_tuning.engagement_decay_keep = 0.0f;
    if (g_tracker_tuning.engagement_decay_keep > 1.0f) g_tracker_tuning.engagement_decay_keep = 1.0f;
    if (g_tracker_tuning.response_ema_keep < 0.0f) g_tracker_tuning.response_ema_keep = 0.0f;
    if (g_tracker_tuning.response_ema_keep > 1.0f) g_tracker_tuning.response_ema_keep = 1.0f;
    if (g_tracker_tuning.response_ema_gain < 0.0f) g_tracker_tuning.response_ema_gain = 0.0f;
    if (g_tracker_tuning.response_ema_gain > 1.0f) g_tracker_tuning.response_ema_gain = 1.0f;
    if (g_tracker_tuning.filter_hold_miss_frames < 0) g_tracker_tuning.filter_hold_miss_frames = 0;
    if (g_tracker_tuning.identity_lock_init_frames < 0) g_tracker_tuning.identity_lock_init_frames = 0;
    if (g_tracker_tuning.identity_jump_dist_px < 0.0f) g_tracker_tuning.identity_jump_dist_px = 0.0f;
    if (g_tracker_tuning.smoothing_alpha_high < 0.0f) g_tracker_tuning.smoothing_alpha_high = 0.0f;
    if (g_tracker_tuning.smoothing_alpha_high > 1.0f) g_tracker_tuning.smoothing_alpha_high = 1.0f;
    if (g_tracker_tuning.smoothing_alpha_low < 0.0f) g_tracker_tuning.smoothing_alpha_low = 0.0f;
    if (g_tracker_tuning.smoothing_alpha_low > 1.0f) g_tracker_tuning.smoothing_alpha_low = 1.0f;
    if (g_tracker_tuning.smoothing_high_confidence_threshold < 0.0f) g_tracker_tuning.smoothing_high_confidence_threshold = 0.0f;
    if (g_tracker_tuning.smoothing_high_confidence_threshold > 1.0f) g_tracker_tuning.smoothing_high_confidence_threshold = 1.0f;
}
int load_tracker_tuning_json(const char *json_path) {

    if (!g_tracker_tuning_initialized) {
        reset_tracker_tuning_defaults();
    }
    if (json_path == NULL) {
        return -1;
    }

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) {
        return -1;
    }

    const cv::FileNode selection = fs["selection"];
    maybe_read_float(selection["confidence_weight"], &g_tracker_tuning.selection_confidence_weight);
    maybe_read_float(selection["age_weight"], &g_tracker_tuning.selection_age_weight);
    maybe_read_float(selection["stability_weight"], &g_tracker_tuning.selection_stability_weight);
    maybe_read_float(selection["continuity_weight"], &g_tracker_tuning.selection_continuity_weight);
    maybe_read_float(selection["engagement_weight"], &g_tracker_tuning.selection_engagement_weight);
    maybe_read_float(selection["age_norm_frames"], &g_tracker_tuning.age_norm_frames);
    maybe_read_float(selection["stability_norm_frames"], &g_tracker_tuning.stability_norm_frames);
    maybe_read_float(selection["missed_norm_frames"], &g_tracker_tuning.missed_norm_frames);
    maybe_read_float(selection["continuity_distance_norm_px"], &g_tracker_tuning.continuity_distance_norm_px);

    const cv::FileNode matching = fs["matching"];
    maybe_read_float(matching["iou_threshold"], &g_tracker_tuning.match_iou_threshold);
    maybe_read_int(matching["stale_track_frames"], &g_tracker_tuning.stale_track_frames);

    const cv::FileNode engagement = fs["engagement"];
    maybe_read_float(engagement["decay_keep"], &g_tracker_tuning.engagement_decay_keep);
    maybe_read_float(engagement["response_diag_scale"], &g_tracker_tuning.response_diag_scale);
    maybe_read_float(engagement["response_motion_weight"], &g_tracker_tuning.response_signal_motion_weight);
    maybe_read_float(engagement["response_confidence_weight"], &g_tracker_tuning.response_signal_confidence_weight);
    maybe_read_float(engagement["response_ema_keep"], &g_tracker_tuning.response_ema_keep);
    maybe_read_float(engagement["response_ema_gain"], &g_tracker_tuning.response_ema_gain);

    const cv::FileNode filter = fs["filter"];
    maybe_read_int(filter["hold_miss_frames"], &g_tracker_tuning.filter_hold_miss_frames);
    maybe_read_int(filter["identity_lock_init_frames"], &g_tracker_tuning.identity_lock_init_frames);
    maybe_read_float(filter["identity_jump_dist_px"], &g_tracker_tuning.identity_jump_dist_px);
    maybe_read_float(filter["identity_confidence_override"], &g_tracker_tuning.identity_confidence_override);
    maybe_read_float(filter["smoothing_alpha_high"], &g_tracker_tuning.smoothing_alpha_high);
    maybe_read_float(filter["smoothing_alpha_low"], &g_tracker_tuning.smoothing_alpha_low);
    maybe_read_float(filter["smoothing_high_confidence_threshold"], &g_tracker_tuning.smoothing_high_confidence_threshold);

    validate_tracker_tuning();

    fprintf(stderr, "Tracker tuning: iou=%.2f stale=%d hold=%d\n",
            g_tracker_tuning.match_iou_threshold, g_tracker_tuning.stale_track_frames, g_tracker_tuning.filter_hold_miss_frames);
    return 0;
}

float clampf(float value, float min_v, float max_v) {
    return (value < min_v) ? min_v : ((value > max_v) ? max_v : value);
}

LaserDotObservation detect_laser_dot(const cv::Mat &frame_bgr) {
    LaserDotObservation observation = {0, cv::Point2f(0.0f, 0.0f), 0.0f};
    if (frame_bgr.empty()) {
        return observation;
    }

    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask_low, mask_high, red_mask;
    cv::inRange(hsv, cv::Scalar(0, 120, 170), cv::Scalar(12, 255, 255), mask_low);
    cv::inRange(hsv, cv::Scalar(170, 120, 170), cv::Scalar(179, 255, 255), mask_high);
    cv::bitwise_or(mask_low, mask_high, red_mask);

    cv::GaussianBlur(red_mask, red_mask, cv::Size(3, 3), 0.0);
    cv::threshold(red_mask, red_mask, 200, 255, cv::THRESH_BINARY);
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float best_score = -1.0f;
    cv::Point2f best_center;
    float best_area = 0.0f;
    for (size_t i = 0; i < contours.size(); ++i) {
        const float area = (float)cv::contourArea(contours[i]);
        if (area < 2.0f || area > 250.0f) {
            continue;
        }

        const float perimeter = (float)cv::arcLength(contours[i], true);
        if (perimeter < 1e-3f) {
            continue;
        }
        const float circularity = 4.0f * 3.1415926f * area / (perimeter * perimeter);
        if (circularity < 0.35f) {
            continue;
        }

        cv::Moments m = cv::moments(contours[i]);
        if (fabs(m.m00) < 1e-5f) {
            continue;
        }

        cv::Point2f c((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
        cv::Point ci((int)c.x, (int)c.y);
        if (ci.x < 0 || ci.y < 0 || ci.x >= frame_bgr.cols || ci.y >= frame_bgr.rows) {
            continue;
        }

        const cv::Vec3b pix = frame_bgr.at<cv::Vec3b>(ci);
        const float red_dominance = (float)pix[2] - 0.5f * ((float)pix[1] + (float)pix[0]);
        if (red_dominance < 25.0f) {
            continue;
        }

        const float score = area * 0.35f + circularity * 80.0f + red_dominance;
        if (score > best_score) {
            best_score = score;
            best_area = area;
            best_center = c;
        }
    }

    if (best_score > 0.0f) {
        observation.detected = 1;
        observation.center = best_center;
        observation.area = best_area;
    }

    return observation;
}

LaserDotObservation stabilize_laser_observation(LaserTrackState *state, const LaserDotObservation &raw) {
    LaserDotObservation out = {0, cv::Point2f(0.0f, 0.0f), 0.0f};
    if (raw.detected) {
        state->stable_detect_frames++;
        state->hold_miss_frames = 0;
        state->last_center = raw.center;
        state->last_area = raw.area;

        if (state->stable_detect_frames >= 2) {
            state->has_valid_lock = 1;
            out = raw;
        }
        return out;
    }

    state->stable_detect_frames = 0;

    if (state->has_valid_lock && state->hold_miss_frames < 6) {
        state->hold_miss_frames++;
        out.detected = 1;
        out.center = state->last_center;
        out.area = state->last_area;
        return out;
    }

    state->has_valid_lock = 0;
    state->hold_miss_frames = 0;
    return out;
}

static float bbox_iou(const Yolov5CatTrackInfo &a, const Yolov5CatTrackInfo &b) {
    const float ax2 = a.x + a.width;
    const float ay2 = a.y + a.height;
    const float bx2 = b.x + b.width;
    const float by2 = b.y + b.height;

    const float ix1 = (a.x > b.x) ? a.x : b.x;
    const float iy1 = (a.y > b.y) ? a.y : b.y;
    const float ix2 = (ax2 < bx2) ? ax2 : bx2;
    const float iy2 = (ay2 < by2) ? ay2 : by2;
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) {
        return 0.0f;
    }

    const float inter = iw * ih;
    const float union_area = a.width * a.height + b.width * b.height - inter;
    if (union_area <= 1e-5f) {
        return 0.0f;
    }
    return inter / union_area;
}

static cv::Point2f bbox_center(const Yolov5CatTrackInfo &b) {
    return cv::Point2f(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
}

static float compute_track_selection_score(const MultiCatTrackerState *state, const MultiCatTrackEntry &track) {
    const float confidence_score = clampf(track.box.confidence, 0.0f, 1.0f);
    const float age_score = clampf((float)track.age_frames / g_tracker_tuning.age_norm_frames, 0.0f, 1.0f);
    const float stability_score =
        clampf((float)track.consecutive_matches / g_tracker_tuning.stability_norm_frames, 0.0f, 1.0f) *
        (1.0f - clampf((float)track.missed_frames / g_tracker_tuning.missed_norm_frames, 0.0f, 1.0f));

    float continuity_score = 0.5f;
    if (state->has_last_active_center) {
        const float d = cv::norm(bbox_center(track.box) - state->last_active_center);
        continuity_score = 1.0f - clampf(d / g_tracker_tuning.continuity_distance_norm_px, 0.0f, 1.0f);
    }

    const float engagement_score = clampf(track.engagement_history, 0.0f, 1.0f);
    return g_tracker_tuning.selection_confidence_weight * confidence_score +
           g_tracker_tuning.selection_age_weight * age_score +
           g_tracker_tuning.selection_stability_weight * stability_score +
           g_tracker_tuning.selection_continuity_weight * continuity_score +
           g_tracker_tuning.selection_engagement_weight * engagement_score;
}

void init_multi_cat_tracker_state(MultiCatTrackerState *state) {
    if (state == NULL) {
        return;
    }
    if (!g_tracker_tuning_initialized) {
        reset_tracker_tuning_defaults();
    }
    state->initialized = 1;
    state->next_track_id = 1;
    state->active_track_id = -1;
    state->has_last_active_center = 0;
    state->last_active_center = cv::Point2f(0.0f, 0.0f);
    for (int i = 0; i < YOLOV5_MAX_CAT_DETECTIONS; ++i) {
        state->tracks[i].active = 0;
        state->tracks[i].track_id = 0;
        state->tracks[i].missed_frames = 0;
        state->tracks[i].age_frames = 0;
        state->tracks[i].consecutive_matches = 0;
        state->tracks[i].engagement_history = 0.0f;
        state->tracks[i].box.has_cat = 0;
        state->tracks[i].box.confidence = 0.0f;
        state->tracks[i].box.x = 0.0f;
        state->tracks[i].box.y = 0.0f;
        state->tracks[i].box.width = 0.0f;
        state->tracks[i].box.height = 0.0f;
    }
}

Yolov5CatTrackInfo update_multi_cat_tracker_and_get_active(MultiCatTrackerState *state, const Yolov5CatDetections *detections) {
    Yolov5CatTrackInfo none = {0, 0, 0, 0, 0, 0};
    if (state == NULL) {
        return none;
    }
    if (!state->initialized) {
        init_multi_cat_tracker_state(state);
    }

    const int det_count = (detections != NULL) ? detections->count : 0;
    int det_used[YOLOV5_MAX_CAT_DETECTIONS] = {0};
    int track_matched[YOLOV5_MAX_CAT_DETECTIONS] = {0};

    for (int ti = 0; ti < YOLOV5_MAX_CAT_DETECTIONS; ++ti) {
        if (!state->tracks[ti].active) continue;
        state->tracks[ti].engagement_history = clampf(state->tracks[ti].engagement_history * g_tracker_tuning.engagement_decay_keep, 0.0f, 1.0f);
    }

    for (int ti = 0; ti < YOLOV5_MAX_CAT_DETECTIONS; ++ti) {
        if (!state->tracks[ti].active) continue;

        float best_iou = 0.0f;
        int best_di = -1;
        for (int di = 0; di < det_count; ++di) {
            if (det_used[di]) continue;
            const float iou = bbox_iou(state->tracks[ti].box, detections->cats[di]);
            if (iou > best_iou) {
                best_iou = iou;
                best_di = di;
            }
        }

        if (best_di >= 0 && best_iou >= g_tracker_tuning.match_iou_threshold) {
            const Yolov5CatTrackInfo prev_box = state->tracks[ti].box;
            const Yolov5CatTrackInfo next_box = detections->cats[best_di];
            state->tracks[ti].box = next_box;
            state->tracks[ti].missed_frames = 0;
            state->tracks[ti].age_frames++;
            state->tracks[ti].consecutive_matches++;

            if (state->tracks[ti].track_id == state->active_track_id) {
                const float motion = cv::norm(bbox_center(next_box) - bbox_center(prev_box));
                const float diag = cv::norm(cv::Point2f(next_box.width, next_box.height));
                const float response = clampf((diag > 1e-3f) ? (motion / (g_tracker_tuning.response_diag_scale * diag + 1e-3f)) : 0.0f, 0.0f, 1.0f);
                const float response_signal =
                    g_tracker_tuning.response_signal_motion_weight * response +
                    g_tracker_tuning.response_signal_confidence_weight * clampf(next_box.confidence, 0.0f, 1.0f);
                state->tracks[ti].engagement_history =
                    clampf(g_tracker_tuning.response_ema_keep * state->tracks[ti].engagement_history +
                               g_tracker_tuning.response_ema_gain * response_signal,
                           0.0f, 1.0f);
            }

            det_used[best_di] = 1;
            track_matched[ti] = 1;
        }
    }

    for (int ti = 0; ti < YOLOV5_MAX_CAT_DETECTIONS; ++ti) {
        if (!state->tracks[ti].active) continue;
        if (track_matched[ti]) continue;
        state->tracks[ti].missed_frames++;
        state->tracks[ti].age_frames++;
        state->tracks[ti].consecutive_matches = 0;
        if (state->tracks[ti].missed_frames > g_tracker_tuning.stale_track_frames) {
            if (state->active_track_id == state->tracks[ti].track_id) {
                state->active_track_id = -1;
                state->has_last_active_center = 0;
                state->last_active_center = cv::Point2f(0.0f, 0.0f);
            }
            state->tracks[ti].active = 0;
        }
    }

    for (int di = 0; di < det_count; ++di) {
        if (det_used[di]) continue;
        int slot = -1;
        int oldest = -1;
        for (int ti = 0; ti < YOLOV5_MAX_CAT_DETECTIONS; ++ti) {
            if (!state->tracks[ti].active) {
                slot = ti;
                break;
            }
            if (state->tracks[ti].missed_frames > oldest) {
                oldest = state->tracks[ti].missed_frames;
                slot = ti;
            }
        }
        state->tracks[slot].active = 1;
        state->tracks[slot].track_id = state->next_track_id++;
        state->tracks[slot].missed_frames = 0;
        state->tracks[slot].age_frames = 1;
        state->tracks[slot].consecutive_matches = 1;
        state->tracks[slot].engagement_history = 0.0f;
        state->tracks[slot].box = detections->cats[di];
    }

    for (int ti = 0; ti < YOLOV5_MAX_CAT_DETECTIONS; ++ti) {
        if (!state->tracks[ti].active) continue;
        if (state->tracks[ti].track_id == state->active_track_id) {
            state->has_last_active_center = 1;
            state->last_active_center = bbox_center(state->tracks[ti].box);
            return state->tracks[ti].box;
        }
    }

    float best_score = -1.0f;
    int best_ti = -1;
    for (int ti = 0; ti < YOLOV5_MAX_CAT_DETECTIONS; ++ti) {
        if (!state->tracks[ti].active) continue;
        const float score = compute_track_selection_score(state, state->tracks[ti]);
        if (score > best_score) {
            best_score = score;
            best_ti = ti;
        }
    }
    if (best_ti >= 0) {
        state->active_track_id = state->tracks[best_ti].track_id;
        state->has_last_active_center = 1;
        state->last_active_center = bbox_center(state->tracks[best_ti].box);
        return state->tracks[best_ti].box;
    }

    state->has_last_active_center = 0;
    return none;
}

Yolov5CatTrackInfo filter_cat_track(CatTrackFilterState *state, const Yolov5CatTrackInfo *raw_track) {
    Yolov5CatTrackInfo out = {0, 0, 0, 0, 0, 0};
    if (raw_track == NULL || !raw_track->has_cat) {
        if (state->initialized && state->hold_miss_frames < g_tracker_tuning.filter_hold_miss_frames) {
            state->hold_miss_frames++;
            return state->filtered;
        }
        state->initialized = 0;
        state->hold_miss_frames = 0;
        state->identity_lock_frames = 0;
        return out;
    }

    state->hold_miss_frames = 0;
    if (!state->initialized) {
        state->filtered = *raw_track;
        state->initialized = 1;
        state->identity_lock_frames = g_tracker_tuning.identity_lock_init_frames;
        return state->filtered;
    }

    const float prev_cx = state->filtered.x + state->filtered.width * 0.5f;
    const float prev_cy = state->filtered.y + state->filtered.height * 0.5f;
    const float new_cx = raw_track->x + raw_track->width * 0.5f;
    const float new_cy = raw_track->y + raw_track->height * 0.5f;
    const float jump_dist = sqrtf((new_cx - prev_cx) * (new_cx - prev_cx) + (new_cy - prev_cy) * (new_cy - prev_cy));

    if (state->identity_lock_frames > 0 &&
        jump_dist > g_tracker_tuning.identity_jump_dist_px &&
        raw_track->confidence < g_tracker_tuning.identity_confidence_override) {
        state->identity_lock_frames--;
        return state->filtered;
    }

    const float alpha =
        (raw_track->confidence > g_tracker_tuning.smoothing_high_confidence_threshold)
            ? g_tracker_tuning.smoothing_alpha_high
            : g_tracker_tuning.smoothing_alpha_low;
    state->filtered.has_cat = 1;
    state->filtered.confidence = alpha * raw_track->confidence + (1.0f - alpha) * state->filtered.confidence;
    state->filtered.x = alpha * raw_track->x + (1.0f - alpha) * state->filtered.x;
    state->filtered.y = alpha * raw_track->y + (1.0f - alpha) * state->filtered.y;
    state->filtered.width = alpha * raw_track->width + (1.0f - alpha) * state->filtered.width;
    state->filtered.height = alpha * raw_track->height + (1.0f - alpha) * state->filtered.height;

    if (jump_dist <= g_tracker_tuning.identity_jump_dist_px ||
        raw_track->confidence >= g_tracker_tuning.identity_confidence_override) {
        state->identity_lock_frames = g_tracker_tuning.identity_lock_init_frames;
    }

    return state->filtered;
}
