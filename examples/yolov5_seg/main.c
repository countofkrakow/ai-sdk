#include <stdio.h>
#include <stdlib.h>

#include <awnn_lib.h>

#include "yolov5_pre_process.h"
#include "yolov5seg_post_process.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <model.nb> <image>\n", argv[0]);
        return -1;
    }

    const char* nbg = argv[1];
    const char* input = argv[2];

    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);

    unsigned int file_size = 0;
    unsigned char* image_data = yolov5_pre_process(input, &file_size);

    void *input_buffers[] = {image_data};
    awnn_set_input_buffers(context, input_buffers);
    awnn_run(context);

    float **results = awnn_get_output_buffers(context);
    yolov5seg_post_process(input, results);

    free(image_data);
    awnn_destroy(context);
    awnn_uninit();
    return 0;
}
