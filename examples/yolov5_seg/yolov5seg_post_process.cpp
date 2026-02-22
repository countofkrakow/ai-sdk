#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "yolov5seg_post_process.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> mask_coeff;
    cv::Mat mask;
};

static inline float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    std::vector<float> areas(objects.size());
    for (size_t i = 0; i < objects.size(); i++)
        areas[i] = objects[i].rect.area();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& a = objects[i];
        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const Object& b = objects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
            {
                keep = 0;
                break;
            }
        }
        if (keep)
            picked.push_back((int)i);
    }
}

static void generate_proposals_yolov5seg(
    int stride,
    const float* feat,
    int feat_h,
    int feat_w,
    float prob_threshold,
    int num_classes,
    int num_masks,
    std::vector<Object>& objects)
{
    static const float anchors[18] = {
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326
    };

    const int anchor_num = 3;
    const int pred_dim = 5 + num_classes + num_masks;
    int anchor_group = stride == 8 ? 1 : (stride == 16 ? 2 : 3);

    for (int h = 0; h < feat_h; h++)
    {
        for (int w = 0; w < feat_w; w++)
        {
            for (int a = 0; a < anchor_num; a++)
            {
                int idx = a * feat_h * feat_w * pred_dim + (h * feat_w + w) * pred_dim;
                const float* p = feat + idx;

                float obj = sigmoid(p[4]);
                if (obj < prob_threshold)
                    continue;

                int class_id = -1;
                float class_score = 0.f;
                for (int c = 0; c < num_classes; c++)
                {
                    float score = sigmoid(p[5 + c]);
                    if (score > class_score)
                    {
                        class_score = score;
                        class_id = c;
                    }
                }

                float score = obj * class_score;
                if (score < prob_threshold)
                    continue;

                float dx = sigmoid(p[0]);
                float dy = sigmoid(p[1]);
                float dw = sigmoid(p[2]);
                float dh = sigmoid(p[3]);

                float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];

                float pred_cx = (dx * 2.f - 0.5f + w) * stride;
                float pred_cy = (dy * 2.f - 0.5f + h) * stride;
                float pred_w = dw * dw * 4.f * anchor_w;
                float pred_h = dh * dh * 4.f * anchor_h;

                Object obj_det;
                obj_det.rect.x = pred_cx - pred_w * 0.5f;
                obj_det.rect.y = pred_cy - pred_h * 0.5f;
                obj_det.rect.width = pred_w;
                obj_det.rect.height = pred_h;
                obj_det.label = class_id;
                obj_det.prob = score;
                obj_det.mask_coeff.resize(num_masks);
                for (int m = 0; m < num_masks; m++)
                    obj_det.mask_coeff[m] = p[5 + num_classes + m];

                objects.push_back(obj_det);
            }
        }
    }
}

static void decode_masks(std::vector<Object>& objects, const float* proto, int proto_h, int proto_w, int proto_c,
                         int img_h, int img_w)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        cv::Mat mask_proto(proto_h, proto_w, CV_32FC1, cv::Scalar(0));

        for (int y = 0; y < proto_h; y++)
        {
            float* dst = mask_proto.ptr<float>(y);
            for (int x = 0; x < proto_w; x++)
            {
                float sum = 0.f;
                int base = (y * proto_w + x) * proto_c; // HWC
                for (int c = 0; c < proto_c; c++)
                    sum += proto[base + c] * objects[i].mask_coeff[c];
                dst[x] = sigmoid(sum);
            }
        }

        cv::Mat full_mask;
        cv::resize(mask_proto, full_mask, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

        objects[i].mask = cv::Mat(img_h, img_w, CV_8UC1, cv::Scalar(0));
        int x0 = std::max(0, (int)objects[i].rect.x);
        int y0 = std::max(0, (int)objects[i].rect.y);
        int x1 = std::min(img_w - 1, (int)(objects[i].rect.x + objects[i].rect.width));
        int y1 = std::min(img_h - 1, (int)(objects[i].rect.y + objects[i].rect.height));

        for (int y = y0; y <= y1; y++)
        {
            const float* src = full_mask.ptr<float>(y);
            unsigned char* dst = objects[i].mask.ptr<unsigned char>(y);
            for (int x = x0; x <= x1; x++)
                dst[x] = src[x] > 0.5f ? 255 : 0;
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    static const unsigned char colors[19][3] = {
        {244, 67, 54}, {233, 30, 99}, {156, 39, 176}, {103, 58, 183}, {63, 81, 181},
        {33, 150, 243}, {3, 169, 244}, {0, 188, 212}, {0, 150, 136}, {76, 175, 80},
        {139, 195, 74}, {205, 220, 57}, {255, 235, 59}, {255, 193, 7}, {255, 152, 0},
        {255, 87, 34}, {121, 85, 72}, {158, 158, 158}, {96, 125, 139}
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        const unsigned char* color = colors[i % 19];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100,
                obj.rect.x, obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        for (int y = 0; y < image.rows; y++)
        {
            const unsigned char* mp = obj.mask.ptr<unsigned char>(y);
            unsigned char* p = image.ptr<unsigned char>(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (mp[x])
                {
                    p[3 * x + 0] = (unsigned char)(p[3 * x + 0] * 0.5f + color[0] * 0.5f);
                    p[3 * x + 1] = (unsigned char)(p[3 * x + 1] * 0.5f + color[1] * 0.5f);
                    p[3 * x + 2] = (unsigned char)(p[3 * x + 2] * 0.5f + color[2] * 0.5f);
                }
            }
        }

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]), 2);
    }

    cv::imwrite("result.png", image);
}

extern "C" {
int yolov5seg_post_process(const char *imagepath, float **output)
{
    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    const int num_classes = 80;
    const int num_masks = 32;
    const int feat_dim = 5 + num_classes + num_masks;

    std::vector<Object> proposals;
    generate_proposals_yolov5seg(8, output[0], 80, 80, 0.25f, num_classes, num_masks, proposals);
    generate_proposals_yolov5seg(16, output[1], 40, 40, 0.25f, num_classes, num_masks, proposals);
    generate_proposals_yolov5seg(32, output[2], 20, 20, 0.25f, num_classes, num_masks, proposals);

    (void)feat_dim;

    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.45f);

    std::vector<Object> objects;
    objects.reserve(picked.size());

    int letterbox_rows = 640;
    int letterbox_cols = 640;
    float scale_letterbox;
    int resize_rows;
    int resize_cols;

    if ((letterbox_rows * 1.0f / bgr.rows) < (letterbox_cols * 1.0f / bgr.cols))
        scale_letterbox = letterbox_rows * 1.0f / bgr.rows;
    else
        scale_letterbox = letterbox_cols * 1.0f / bgr.cols;

    resize_cols = int(scale_letterbox * bgr.cols);
    resize_rows = int(scale_letterbox * bgr.rows);

    int pad_h = (letterbox_rows - resize_rows) / 2;
    int pad_w = (letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)bgr.rows / resize_rows;
    float ratio_y = (float)bgr.cols / resize_cols;

    for (size_t i = 0; i < picked.size(); i++)
    {
        Object obj = proposals[picked[i]];

        float x0 = (obj.rect.x - pad_w) * ratio_x;
        float y0 = (obj.rect.y - pad_h) * ratio_y;
        float x1 = (obj.rect.x + obj.rect.width - pad_w) * ratio_x;
        float y1 = (obj.rect.y + obj.rect.height - pad_h) * ratio_y;

        x0 = std::max(std::min(x0, (float)(bgr.cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(bgr.rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(bgr.cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(bgr.rows - 1)), 0.f);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;

        objects.push_back(obj);
    }

    // Typical yolov5-seg proto tensor is 160x160x32 for 640 input.
    decode_masks(objects, output[3], 160, 160, num_masks, bgr.rows, bgr.cols);

    fprintf(stderr, "seg detection num: %zu\n", objects.size());
    draw_objects(bgr, objects);
    return 0;
}
}
