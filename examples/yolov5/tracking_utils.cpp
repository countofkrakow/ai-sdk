#include "tracking_utils.h"

#include <math.h>
#include <vector>
#include <opencv2/imgproc.hpp>

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
        // Circularity near 1.0 indicates a dot-like blob.
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
        // Additional robustness under bright scenes: require red channel dominance.
        const float red_dominance = (float)pix[2] - 0.5f * ((float)pix[1] + (float)pix[0]);
        if (red_dominance < 25.0f) {
            continue;
        }

        // Prefer small bright circular red blobs.
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
            out = raw;
        }
        return out;
    }

    state->stable_detect_frames = 0;
    if (state->hold_miss_frames < 6) {
        state->hold_miss_frames++;
        out.detected = 1;
        out.center = state->last_center;
        out.area = state->last_area;
    }

    return out;
}

Yolov5CatTrackInfo filter_cat_track(CatTrackFilterState *state, const Yolov5CatTrackInfo *raw_track) {
    Yolov5CatTrackInfo out = {0, 0, 0, 0, 0, 0};
    if (raw_track == NULL || !raw_track->has_cat) {
        if (state->initialized && state->hold_miss_frames < 12) {
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
        state->identity_lock_frames = 10;
        return state->filtered;
    }

    // Better multi-cat identity handling:
    // If detection suddenly jumps far from the current track and confidence is
    // not strong, keep the previous target briefly to avoid rapid cat-switching.
    const float prev_cx = state->filtered.x + state->filtered.width * 0.5f;
    const float prev_cy = state->filtered.y + state->filtered.height * 0.5f;
    const float new_cx = raw_track->x + raw_track->width * 0.5f;
    const float new_cy = raw_track->y + raw_track->height * 0.5f;
    const float jump_dist = sqrtf((new_cx - prev_cx) * (new_cx - prev_cx) + (new_cy - prev_cy) * (new_cy - prev_cy));

    if (state->identity_lock_frames > 0 && jump_dist > 120.0f && raw_track->confidence < 0.78f) {
        state->identity_lock_frames--;
        return state->filtered;
    }

    const float alpha = (raw_track->confidence > 0.7f) ? 0.35f : 0.2f;
    state->filtered.has_cat = 1;
    state->filtered.confidence = alpha * raw_track->confidence + (1.0f - alpha) * state->filtered.confidence;
    state->filtered.x = alpha * raw_track->x + (1.0f - alpha) * state->filtered.x;
    state->filtered.y = alpha * raw_track->y + (1.0f - alpha) * state->filtered.y;
    state->filtered.width = alpha * raw_track->width + (1.0f - alpha) * state->filtered.width;
    state->filtered.height = alpha * raw_track->height + (1.0f - alpha) * state->filtered.height;

    if (jump_dist <= 120.0f || raw_track->confidence >= 0.78f) {
        state->identity_lock_frames = 10;
    }

    return state->filtered;
}
