#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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

    const int input_channels = 3;
    const int input_height = 480;
    const int input_width = 640;

    cv::VideoCapture camera(camera_device, cv::CAP_V4L2);
    if (!camera.isOpened()) {
        fprintf(stderr, "Failed to open webcam device: %s\n", camera_device);
        return -1;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, input_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, input_height);

    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);
    if (context == NULL) {
        fprintf(stderr, "Failed to create NPU context with nbg: %s\n", nbg);
        camera.release();
        awnn_uninit();
        return -1;
    }

    printf("Running live detection from %s\n", camera_device);
    printf("Enforcing input shape CxHxW = %dx%dx%d\n", input_channels, input_height, input_width);
    printf("Annotated detections will be written to result.png\n");
    printf("Displaying live detections in OpenCV window (press q to quit)\n");

    int printed_resolution = 0;

    while (1) {
        cv::Mat raw_frame;
        if (!camera.read(raw_frame) || raw_frame.empty()) {
            fprintf(stderr, "Failed to read frame from webcam\n");
            usleep(100000);
            continue;
        }

        if (!printed_resolution) {
            printf("Webcam frame resolution: %dx%d\n", raw_frame.cols, raw_frame.rows);
            printed_resolution = 1;
        }

        cv::Mat frame = raw_frame;
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        if (frame.cols != input_width || frame.rows != input_height) {
            cv::resize(frame, frame, cv::Size(input_width, input_height));
        }

        if (frame.channels() != input_channels) {
            fprintf(stderr, "Unexpected channel count after normalization: %d\n", frame.channels());
            usleep(100000);
            continue;
        }

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
        awnn_run(context);

        float **results = awnn_get_output_buffers(context);
        yolov5_post_process(frame_file, results);

        cv::Mat detection = cv::imread("result.png");
        if (!detection.empty()) {
            cv::imshow("YOLOv5 Live Detection", detection);
        }

        free(plant_data);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }

        usleep(30000);
    }

    awnn_destroy(context);
    awnn_uninit();
    camera.release();
    cv::destroyAllWindows();

    return 0;
}
