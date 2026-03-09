#ifndef __YOLOV5_POST_PROCESS_H__
#define __YOLOV5_POST_PROCESS_H__
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Yolov5CatTrackInfo {
    int has_cat;
    float confidence;
    float x;
    float y;
    float width;
    float height;
} Yolov5CatTrackInfo;

int yolov5_post_process(const char *imagepath, float **output, Yolov5CatTrackInfo *track_info);

#ifdef __cplusplus
}
#endif

#endif
