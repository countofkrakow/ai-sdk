#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <linux/videodev2.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <awnn_lib.h>

#include "image_utils.h"
#include "yolov5_pre_process.h"
#include "yolov5_post_process.h"

struct CameraBuffer {
    void *start;
    size_t length;
};

static int xioctl(int fd, int request, void *arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

int main(int argc, char **argv) {
    printf("%s nbg [camera_device]\n", argv[0]);
    if (argc < 2) {
        printf("Arguments count %d is incorrect!\n", argc);
        return -1;
    }

    const char *nbg = argv[1];
    const char *camera_device = (argc >= 3) ? argv[2] : "/dev/video0";
    const char *frame_file = "live_frame.jpg";

    int camera_fd = open(camera_device, O_RDWR);
    if (camera_fd < 0) {
        fprintf(stderr, "Failed to open webcam device %s: %s\n", camera_device, strerror(errno));
        return -1;
    }

    struct v4l2_capability capability;
    memset(&capability, 0, sizeof(capability));
    if (xioctl(camera_fd, VIDIOC_QUERYCAP, &capability) < 0) {
        fprintf(stderr, "VIDIOC_QUERYCAP failed: %s\n", strerror(errno));
        close(camera_fd);
        return -1;
    }

    if (!(capability.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        !(capability.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "Device %s does not support V4L2 capture/streaming\n", camera_device);
        close(camera_fd);
        return -1;
    }

    struct v4l2_format format;
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = 640;
    format.fmt.pix.height = 480;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(camera_fd, VIDIOC_S_FMT, &format) < 0) {
        fprintf(stderr, "VIDIOC_S_FMT failed: %s\n", strerror(errno));
        close(camera_fd);
        return -1;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(camera_fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        fprintf(stderr, "VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        close(camera_fd);
        return -1;
    }

    CameraBuffer *buffers = (CameraBuffer *)calloc(req.count, sizeof(CameraBuffer));
    if (buffers == NULL) {
        fprintf(stderr, "Failed to allocate camera buffers metadata\n");
        close(camera_fd);
        return -1;
    }

    for (unsigned int i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(camera_fd, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "VIDIOC_QUERYBUF failed: %s\n", strerror(errno));
            free(buffers);
            close(camera_fd);
            return -1;
        }

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, camera_fd, buf.m.offset);

        if (buffers[i].start == MAP_FAILED) {
            fprintf(stderr, "mmap failed: %s\n", strerror(errno));
            free(buffers);
            close(camera_fd);
            return -1;
        }

        if (xioctl(camera_fd, VIDIOC_QBUF, &buf) < 0) {
            fprintf(stderr, "VIDIOC_QBUF failed: %s\n", strerror(errno));
            free(buffers);
            close(camera_fd);
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(camera_fd, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "VIDIOC_STREAMON failed: %s\n", strerror(errno));
        free(buffers);
        close(camera_fd);
        return -1;
    }

    // npu init
    awnn_init();
    // create network
    Awnn_Context_t *context = awnn_create(nbg);

    if (context == NULL) {
        fprintf(stderr, "Failed to create NPU context with nbg: %s\n", nbg);
        xioctl(camera_fd, VIDIOC_STREAMOFF, &type);
        for (unsigned int i = 0; i < req.count; ++i) {
            munmap(buffers[i].start, buffers[i].length);
        }
        free(buffers);
        close(camera_fd);
        awnn_uninit();
        return -1;
    }

    printf("Running live detection from %s\n", camera_device);
    printf("Annotated detections will be written to result.png\n");

    int printed_resolution = 0;

    while (1) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (xioctl(camera_fd, VIDIOC_DQBUF, &buf) < 0) {
            fprintf(stderr, "VIDIOC_DQBUF failed: %s\n", strerror(errno));
            usleep(100000);
            continue;
        }

        cv::Mat yuyv(format.fmt.pix.height, format.fmt.pix.width, CV_8UC2, buffers[buf.index].start);
        cv::Mat frame;
        cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUYV);

        if (!printed_resolution) {
            printf("Webcam frame resolution: %dx%d\n", frame.cols, frame.rows);
            printed_resolution = 1;
        }

        if (!cv::imwrite(frame_file, frame)) {
            fprintf(stderr, "Failed to write frame image: %s\n", frame_file);
            xioctl(camera_fd, VIDIOC_QBUF, &buf);
            usleep(100000);
            continue;
        }

        unsigned int file_size = 0;
        unsigned char *plant_data = yolov5_pre_process(frame_file, &file_size);
        if (plant_data == NULL) {
            fprintf(stderr, "Pre-process failed for frame: %s\n", frame_file);
            xioctl(camera_fd, VIDIOC_QBUF, &buf);
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

        if (xioctl(camera_fd, VIDIOC_QBUF, &buf) < 0) {
            fprintf(stderr, "VIDIOC_QBUF failed: %s\n", strerror(errno));
            usleep(100000);
            continue;
        }

        // Small delay to avoid tight spin loop and reduce CPU usage.
        usleep(30000);
    }

    // Unreachable in normal usage, but left for completeness.
    awnn_destroy(context);
    awnn_uninit();

    xioctl(camera_fd, VIDIOC_STREAMOFF, &type);
    for (unsigned int i = 0; i < req.count; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    free(buffers);
    close(camera_fd);

    return 0;
}
