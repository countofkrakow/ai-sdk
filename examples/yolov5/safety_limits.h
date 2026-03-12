#ifndef YOLOV5_SAFETY_LIMITS_H_
#define YOLOV5_SAFETY_LIMITS_H_

static const float kPanClampMinDeg = -45.0f;
static const float kPanClampMaxDeg = 45.0f;
static const float kTiltClampMinDeg = -45.0f;
static const float kTiltClampMaxDeg = 45.0f;

static const float kDeadmanCameraTimeoutSec = 2.0f;

#endif
