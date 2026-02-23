#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <awnn_lib.h>

#include "image_utils.h"
#include "yolov5_pre_process.h"
#include "yolov5_post_process.h"

int main(int argc, char **argv) {
    printf("%s nbg [camera_device]\n", argv[0]);
    if (argc < 2) {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }

    const char *nbg = argv[1];
    const char *camera_device = (argc >= 3) ? argv[2] : "/dev/video0";
    const char *frame_file = "live_frame.jpg";

    cv::VideoCapture camera(camera_device, cv::CAP_V4L2);
    if (!camera.isOpened()) {
        fprintf(stderr, "Failed to open webcam device: %s\n", camera_device);
        return -1;
    }

    // npu init
    awnn_init();
    // create network
    Awnn_Context_t *context = awnn_create(nbg);

    if (context == NULL) {
        fprintf(stderr, "Failed to create NPU context with nbg: %s\n", nbg);
        camera.release();
        awnn_uninit();
        return -1;
    }

    printf("Running live detection from %s\n", camera_device);
    printf("Annotated detections will be written to result.png\n");

    int printed_resolution = 0;

    while (1) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            fprintf(stderr, "Failed to read frame from webcam\n");
            usleep(100000);
            continue;
        }

        if (!printed_resolution) {
            printf("Webcam frame resolution: %dx%d\n", frame.cols, frame.rows);
            printed_resolution = 1;
        }

        // Save the most recent frame so existing preprocess/postprocess can use file path input.
        if (!cv::imwrite(frame_file, frame)) {
            fprintf(stderr, "Failed to write frame image: %s\n", frame_file);
            usleep(100000);
            continue;
        }

        unsigned int file_size = 0;
        unsigned char *plant_data = yolov5_pre_process(frame_file, &file_size);
        if (plant_data == NULL) {
            fprintf(stderr, "Pre-process failed for frame: %s\n", frame_file);
            usleep(100000);
            continue;
        }

        void *input_buffers[] = {plant_data};
        awnn_set_input_buffers(context, input_buffers);

        // process network
        awnn_run(context);

        // get result
        float **results = awnn_get_output_buffers(context);

        // post process (writes annotated image to result.png)
        yolov5_post_process(frame_file, results);

        free(plant_data);

        // Small delay to avoid tight spin loop and reduce CPU usage.
        usleep(30000);
    }

    // Unreachable in normal usage, but left for completeness.
    awnn_destroy(context);
    awnn_uninit();
    camera.release();

    return 0;
}
