#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <awnn_lib.h>

#include "yolov5_pre_process.h"
#include "yolov5_post_process.h"

int main(int argc, char **argv) {
    printf("%s nbg [camera_index]\n", argv[0]);
    if (argc < 2) {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }

    const char *nbg = argv[1];
    int camera_index = 0;
    if (argc >= 3) {
        camera_index = atoi(argv[2]);
    }

    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open USB webcam at index %d\n", camera_index);
        return -1;
    }

    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);
    if (context == NULL) {
        fprintf(stderr, "Failed to create AWNN context\n");
        awnn_uninit();
        return -1;
    }

    const std::string latest_frame_path = "/tmp/yolov5_latest_frame.jpg";
    cv::namedWindow("yolov5 detections", cv::WINDOW_AUTOSIZE);

    while (1) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            fprintf(stderr, "Failed to read frame from webcam\n");
            continue;
        }

        if (!cv::imwrite(latest_frame_path, frame)) {
            fprintf(stderr, "Failed to save latest webcam frame\n");
            continue;
        }

        unsigned int file_size = 0;
        unsigned char *input_data = yolov5_pre_process(latest_frame_path.c_str(), &file_size);
        if (input_data == NULL) {
            fprintf(stderr, "Pre-process failed\n");
            continue;
        }

        void *input_buffers[] = {input_data};
        awnn_set_input_buffers(context, input_buffers);
        awnn_run(context);
        float **results = awnn_get_output_buffers(context);

        yolov5_post_process(latest_frame_path.c_str(), results);

        free(input_data);

        cv::Mat detections = cv::imread("result.png", 1);
        if (!detections.empty()) {
            cv::imshow("yolov5 detections", detections);
        }

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    awnn_destroy(context);
    awnn_uninit();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
