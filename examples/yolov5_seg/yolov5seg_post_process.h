#ifndef __YOLOV5SEG_POST_PROCESS_H__
#define __YOLOV5SEG_POST_PROCESS_H__
#ifdef __cplusplus
extern "C" {
#endif

int yolov5seg_post_process(const char *imagepath, float **output);

#ifdef __cplusplus
}
#endif

#endif
