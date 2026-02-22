#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <awnn_lib.h>

#include "yolov5_pre_process.h"
#include "yolov5_post_process.h"

static bool fetch_latest_frame(cv::VideoCapture &capture, cv::Mat &frame, int drain_grabs)
{
    if (!capture.isOpened()) {
        return false;
    }

    for (int i = 0; i < drain_grabs; ++i) {
        if (!capture.grab()) {
            break;
        }
    }

    return capture.read(frame) && !frame.empty();
}

int main(int argc, char **argv) {
    printf("%s nbg [device_index] [output_path] [drain_grabs]\n", argv[0]);
    printf("Press 'q' or ESC to quit.\n");
    if (argc < 2) {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }

    const char *nbg = argv[1];
    int device_index = argc >= 3 ? atoi(argv[2]) : 0;
    const char *output_path = argc >= 4 ? argv[3] : "result.png";
    int drain_grabs = argc >= 5 ? atoi(argv[4]) : 4;

    const char *tmp_input_path = "/tmp/yolov5_webcam_input.jpg";

    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);
    if (context == NULL) {
        fprintf(stderr, "Failed to create AWNN context\n");
        awnn_uninit();
        return -1;
    }

    cv::VideoCapture capture(device_index, cv::CAP_V4L2);
    if (!capture.isOpened()) {
        fprintf(stderr, "Failed to open USB webcam /dev/video%d\n", device_index);
        awnn_destroy(context);
        awnn_uninit();
        return -1;
    }

    capture.set(cv::CAP_PROP_BUFFERSIZE, 1);

    const char *window_name = "yolov5_webcam";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    int frame_id = 0;
    while (1) {
        cv::Mat frame;
        if (!fetch_latest_frame(capture, frame, drain_grabs)) {
            fprintf(stderr, "Failed to read latest webcam frame\n");
            usleep(10000);
            continue;
        }

        if (!cv::imwrite(tmp_input_path, frame)) {
            fprintf(stderr, "Failed to write temporary frame %s\n", tmp_input_path);
            continue;
        }

        unsigned int file_size = 0;
        unsigned char *input_data = yolov5_pre_process(tmp_input_path, &file_size);
        if (input_data == NULL) {
            fprintf(stderr, "Pre-process failed for frame %d\n", frame_id);
            continue;
        }

        void *input_buffers[] = {input_data};
        awnn_set_input_buffers(context, input_buffers);
        awnn_run(context);

        float **results = awnn_get_output_buffers(context);
        if (yolov5_post_process(tmp_input_path, results) != 0) {
            fprintf(stderr, "Post-process failed for frame %d\n", frame_id);
            free(input_data);
            continue;
        }

        cv::Mat annotated = cv::imread("result.png", 1);
        if (!annotated.empty()) {
            cv::imshow(window_name, annotated);
            if (output_path[0] != '\0') {
                cv::imwrite(output_path, annotated);
            }
        }

        free(input_data);
        ++frame_id;

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    cv::destroyWindow(window_name);
    capture.release();
    awnn_destroy(context);
    awnn_uninit();
    return 0;
}
