#ifndef __YOLOV5_POST_PROCESS_H__
#define __YOLOV5_POST_PROCESS_H__
#ifdef __cplusplus
extern "C" {
#endif

#define YOLOV5_MAX_SCENE_CATS 8
#define YOLOV5_MAX_SCENE_PEOPLE 4

typedef struct Yolov5CatTrackInfo {
    int has_cat;
    float confidence;
    float x;
    float y;
    float width;
    float height;
} Yolov5CatTrackInfo;

typedef struct Yolov5TrackedBox {
    float confidence;
    float x;
    float y;
    float width;
    float height;
} Yolov5TrackedBox;

typedef struct Yolov5SceneDetections {
    int cat_count;
    Yolov5TrackedBox cats[YOLOV5_MAX_SCENE_CATS];
    int person_count;
    Yolov5TrackedBox people[YOLOV5_MAX_SCENE_PEOPLE];
} Yolov5SceneDetections;

int yolov5_post_process(
    const char *imagepath,
    float **output,
    Yolov5CatTrackInfo *track_info,
    Yolov5SceneDetections *scene_detections);

#ifdef __cplusplus
}
#endif

#endif
