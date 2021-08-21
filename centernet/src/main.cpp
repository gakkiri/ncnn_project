#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>

#define input_h 448
#define input_w 672
#define pool_size 3
#define conf_th 0.3

struct Box
{
    float x1;
    float y1;
    float x2;
    float y2;
};
 
struct Detection
{
    Box bbox;
    int classId;
    float confidence;
};


std::vector<std::string> coco_class_name = {
     "person", "bicycle", "car", "motorcycle", "airplane",
     "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
     "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
     "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
     "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
     "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
     "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
     "scissors", "teddy bear", "hair drier", "toothbrush"};


void decode(ncnn::Mat& cls_mat, ncnn::Mat &wh_mat, ncnn::Mat &reg_mat, std::vector<Detection> &results) {
    float *hm  = (float*)cls_mat.data;  // c * h * w
    float *wh  = (float*)wh_mat.data;
    float *reg = (float*)reg_mat.data;
    const int w = cls_mat.w;
    const int h = cls_mat.h;
    const int c = cls_mat.c;

    for (int i = 0; i < w*h*c; i++) {
        if (i >= w*h*c) return;

        int grid_x = i % w;
        int grid_y = (i / w) % h;
        int cls = i / w / h;
        int reg_index = i - cls*w*h;
        float confidence = hm[i];

        float cx, cy;
        if (confidence > conf_th) {
            Detection det;

            cx = grid_x + reg[reg_index];
            cy = grid_y +reg[reg_index + w*h];

            det.bbox.x1 = (cx - wh[reg_index] / 2) * 4;
            det.bbox.y1 = (cy - wh[reg_index + w*h] / 2) * 4;
            det.bbox.x2 = (cx + wh[reg_index] / 2) * 4;
            det.bbox.y2 = (cy + wh[reg_index + w*h] / 2) * 4;
            det.classId = cls;
            det.confidence = confidence;
            results.push_back(det);
        }
    }
}

void rescale_box(std::vector<Detection> &results, const cv::Mat &raw_bgr) {
    const int raw_h = raw_bgr.rows;
    const int raw_w = raw_bgr.cols;

    float scale_h = (float)raw_h / (float)input_h;
    float scale_w = (float)raw_w / (float)input_w;
    float dx = ((float)input_w - (float)raw_w/scale_w) / 2;
    float dy = ((float)input_h - (float)raw_h/scale_h) / 2;

    for (auto &result : results) {
        float x1 = (result.bbox.x1 - dx) * scale_w;
        float y1 = (result.bbox.y1 - dy) * scale_h;
        float x2 = (result.bbox.x2 - dx) * scale_w;
        float y2 = (result.bbox.y2 - dy) * scale_h;

        x1 = (x1 > 0) ? x1 : 0;
        y1 = (y1 > 0) ? y1 : 0;
        x2 = (x2 < raw_w) ? x2 : raw_w - 1;
        y2 = (y2 < raw_h) ? y2 : raw_h - 1;
        result.bbox.x1 = x1;
        result.bbox.y1 = y1;
        result.bbox.x2 = x2;
        result.bbox.y2 = y2;
    }
}

void draw(const std::vector<Detection> &results, cv::Mat &img, const std::vector<cv::Scalar> &colors)
{   
    for (const auto &result : results)
    {
        const auto &cls = result.classId;
        const auto &x1  = result.bbox.x1;
        const auto &y1  = result.bbox.y1;
        const auto &x2  = result.bbox.x2;
        const auto &y2  = result.bbox.y2;
        const auto &con = result.confidence;

        std::string text = "class: " + std::to_string(cls) + "  score: "  + std::to_string(con);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colors[cls], 3);
        cv::putText(img, text, cv::Point(x1, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5, cv::LINE_AA);
    }
}


int main(int argc, char** argv) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s [imagepath] [weightpath]\n", argv[0]);
        return -1;
    }

    std::cout << ">>> prepare data..." << std::endl;

    const char* imagepath = argv[1];
    cv::Mat bgr = cv::imread(imagepath, 1);  // c, h, w
    if (bgr.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    const char* weight_root = argv[2];
    std::string const& path_param = std::string(weight_root) + "/ctnet.param";
    std::string const& path_bin = std::string(weight_root) + "/ctnet.bin";

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, input_w, input_h);
    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);  // h, w, c

    std::cout << ">>> input shape ==> " << " h: " << in.h << " w: " << in.w << " c: " << in.c << std::endl;
    std::cout << ">>> forward..." << std::endl;

    ncnn::Net ctnet;
    ctnet.opt.use_vulkan_compute = true;  // gpu
    ctnet.load_param("../../ctnet.param");
    ctnet.load_model("../../ctnet.bin");
    ncnn::Extractor ex = ctnet.create_extractor();

    ex.input("input", in);
    ncnn::Mat cls, wh, reg;
    ex.extract("cls", cls);
    ex.extract("wh", wh);
    ex.extract("reg", reg);

    std::cout << ">>> cls ==> h: " << cls.h << "  w: " << cls.w << "  c: " << cls.c << std::endl;
    std::cout << ">>> wh  ==> h: " << wh.h << "  w: " << wh.w << "  c: " << wh.c << std::endl;
    std::cout << ">>> reg ==> h: " << reg.h << "  w: " << reg.w << "  c: " << reg.c << std::endl;

    std::cout << ">>> postprocessing..." << std::endl;

    std::cout << ">>> pseudo nms..." << std::endl;
    ncnn::Layer* pseudo_nms = ncnn::create_layer("Pooling");
    ncnn::ParamDict pd;
    ncnn::Option opt;
    int pad = (pool_size - 1) / 2;
    pd.set(1, pool_size);  // kernel size is 3
    pd.set(3, pad);
    pseudo_nms->load_param(pd);
    pseudo_nms->create_pipeline(opt);

    ncnn::Mat cls_max;
    pseudo_nms->forward(cls, cls_max, opt);
    std::cout << ">>> cls_max ==> h: " << cls_max.h << "  w: " << cls_max.w << "  c: " << cls_max.c << std::endl;
    {
        float *cls_data = (float*)cls.data;
        float *clsmax_data = (float*)cls_max.data;
        for (int i = 0; i < cls.h; i++) {
            for (int j  = 0; j < cls.w; j++) {
                for (int k = 0; k < cls.c; k++) {
                    int pos = k*cls.w*cls.h + i*cls.w + j;
                    if (cls_data[pos] != clsmax_data[pos]) {
                        cls_data[pos] = 0;
                    }
                }
            }
        }
    }
    std::cout << ">>> pseudo done!" << std::endl;

    std::cout << ">>> get result..." << std::endl;
    std::vector<Detection> results;
    decode(cls, wh, reg, results);
    std::cout << ">>> found " << results.size() << " objects!" << std::endl;
    rescale_box(results, bgr);

    cv::RNG rng(time(0));
    std::vector<cv::Scalar> colors(80);
    for (int  i = 0; i < 80; ++i) {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
    draw(results, bgr, colors);

    std::cout << ">>> done!";

    cv::imshow("", bgr);
    cv::waitKey(0);

    return 0;
}
