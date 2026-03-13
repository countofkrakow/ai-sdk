#ifndef __YOLOV5_POST_PROCESS_H__
#define __YOLOV5_POST_PROCESS_H__
#ifdef __cplusplus
extern "C" {
#endif

// Compact cat bbox used by downstream tracking/control.
typedef struct Yolov5CatTrackInfo {
    int has_cat;
    float confidence;
    float x;
    float y;
    float width;
    float height;
} Yolov5CatTrackInfo;

#define YOLOV5_MAX_CAT_DETECTIONS 8

typedef struct Yolov5CatDetections {
    int count;
    Yolov5CatTrackInfo cats[YOLOV5_MAX_CAT_DETECTIONS];
} Yolov5CatDetections;

int yolov5_post_process(const char *imagepath, float **output, Yolov5CatTrackInfo *track_info, Yolov5CatDetections *cat_detections);

#ifdef __cplusplus
}
#endif

#endif
