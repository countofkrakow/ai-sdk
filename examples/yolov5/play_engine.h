#ifndef YOLOV5_PLAY_ENGINE_H_
#define YOLOV5_PLAY_ENGINE_H_

#include <opencv2/core.hpp>
#include "play_algorithms.h"

struct PlayEngine;

struct PlayEngineStepInput {
    Yolov5CatTrackInfo cat;
    float detection_confidence;
    cv::Point2f virtual_laser_point;
    int frame_index;
    int frame_w;
    int frame_h;
    float dt_sec;
};

struct PlayEngineStepOutput {
    cv::Point2f play_target;
    const char *algo_name;
    float engagement_score;
    enum PlayDirectorIntent director_intent;
};

PlayEngine *play_engine_init();
void play_engine_reset(PlayEngine *engine);
int play_engine_step(PlayEngine *engine, const PlayEngineStepInput *in, PlayEngineStepOutput *out);
void play_engine_destroy(PlayEngine *engine);

#endif
