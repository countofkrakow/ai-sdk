#include "play_engine.h"

#include <stdlib.h>

struct PlayEngine {
    CatPlayState state;
};

PlayEngine *play_engine_init() {
    PlayEngine *engine = (PlayEngine *)calloc(1, sizeof(PlayEngine));
    if (engine == NULL) {
        return NULL;
    }
    init_cat_play_state(&engine->state);
    return engine;
}

void play_engine_reset(PlayEngine *engine) {
    if (engine == NULL) return;
    init_cat_play_state(&engine->state);
}

int play_engine_step(PlayEngine *engine, const PlayEngineStepInput *in, PlayEngineStepOutput *out) {
    if (engine == NULL || in == NULL || out == NULL) {
        return -1;
    }

    const char *algo_name = "unknown";
    const cv::Point2f target = build_cat_play_target(
        &engine->state,
        in->cat,
        in->detection_confidence,
        in->virtual_laser_point,
        in->frame_index,
        in->frame_w,
        in->frame_h,
        in->dt_sec,
        &algo_name);

    out->play_target = target;
    out->algo_name = algo_name;
    out->engagement_score = engine->state.engagement_score;
    out->director_intent = engine->state.director_intent;
    return 0;
}

void play_engine_destroy(PlayEngine *engine) {
    if (engine != NULL) {
        free(engine);
    }
}
