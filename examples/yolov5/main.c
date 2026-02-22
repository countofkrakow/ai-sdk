#include <stdio.h>
#include <stdlib.h>

#include <awnn_lib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "yolov5_pre_process.h"
#include "yolov5_post_process.h"

int main(int argc, char **argv) {
    printf("%s nbg [camera_index] [frame_path]\n", argv[0]);
    if (argc < 2)
    {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }

    const char* nbg = argv[1];
    int camera_index = (argc >= 3) ? atoi(argv[2]) : 0;
    const char* frame_path = (argc >= 4) ? argv[3] : "/tmp/yolov5_webcam_frame.jpg";

    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open webcam index %d\n", camera_index);
        return -1;
    }

    // npu init
    awnn_init();
    // create network
    Awnn_Context_t *context = awnn_create(nbg);

    cv::namedWindow("yolov5_webcam", cv::WINDOW_AUTOSIZE);

    while (1) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            fprintf(stderr, "Failed to capture frame from webcam.\n");
            break;
        }

        if (!cv::imwrite(frame_path, frame)) {
            fprintf(stderr, "Failed to save frame to %s\n", frame_path);
            break;
        }

        unsigned int file_size = 0;
        unsigned char* plant_data = yolov5_pre_process(frame_path, &file_size);
        if (!plant_data) {
            fprintf(stderr, "Pre-process failed.\n");
            break;
        }

        void *input_buffers[] = {plant_data};
        awnn_set_input_buffers(context, input_buffers);
        awnn_run(context);

        float **results = awnn_get_output_buffers(context);
        yolov5_post_process(frame_path, results);

        free(plant_data);

        cv::Mat vis = cv::imread("result.png", 1);
        if (vis.empty()) {
            fprintf(stderr, "Failed to load result.png for display\n");
            break;
        }

        cv::imshow("yolov5_webcam", vis);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    cap.release();

    awnn_destroy(context);
    awnn_uninit();

    return 0;
}
