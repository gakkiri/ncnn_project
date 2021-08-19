#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>

#define input_h 512
#define input_w 1024
#define alpha 0.7


const std::vector<std::vector<int>> colors = {
    {56, 0, 255}, {226, 255, 0}, {0, 94, 255}, {0, 37, 255}, {0, 255, 94},
    {255, 226, 0}, {0, 18, 255}, {255, 151, 0}, {170, 0, 255}, {0, 255, 56},
    {255, 0, 75}, {0, 75, 255}, {0, 255, 169}, {255, 0, 207}, {75, 255, 0},
    {207, 0, 255}, {37, 0, 255}, {0, 207, 255}, {94, 0, 255}
};


int main(int argc, char** argv) {

    if (argc != 2) {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    std::cout << ">>> prepare data..." << std::endl;

    const char* imagepath = argv[1];
    cv::Mat bgr = cv::imread(imagepath, 1);  // c, h, w
    if (bgr.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, input_w, input_h);

    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);  // h, w, c

    std::cout << ">>> forward..." << std::endl;

    ncnn::Net gscnn;
    gscnn.opt.use_vulkan_compute = true;
    gscnn.load_param("weight/ckp_512_1024.param");
    gscnn.load_model("weight/ckp_512_1024.bin");
    ncnn::Extractor ex = gscnn.create_extractor();

    ex.input("input", in);
    ncnn::Mat final_seg;
    ex.extract("final_seg", final_seg);

    std::cout << ">>> postprocessing..." << std::endl;

    cv::Mat bitmap = cv::Mat::zeros(final_seg.h, final_seg.w, CV_8UC3);
    std::cout << "h: " << bitmap.rows << "  w: " << bitmap.cols << "  c: " << bitmap.channels() << std::endl;

    float *srcdata = (float*)final_seg.data;  // (512, 1024, 19)
    // unsigned char *data = bitmap.data;
    // auto *data = bitmap.ptr<uchar>();

    for (int i = 0; i < final_seg.h; i++) {
        for (int j = 0; j < final_seg.w; j++) {
            float tmp = srcdata[i*final_seg.w + j];
            int maxk = 0;
            for (int k = 0; k < final_seg.c; k++) {
                if (tmp < srcdata[k*final_seg.w*final_seg.h + i*final_seg.w + j]) {
                    tmp = srcdata[k*final_seg.w*final_seg.h + i*final_seg.w + j];
                    maxk = k;
                }
            }

            auto *p = bitmap.ptr<uchar>(i, j);
            for (int k = 0; k < 3; k++) {
                p[k] = cv::saturate_cast<uchar>(colors[maxk][k]);
            }
        }
    }
    cv::Mat resize_mask;
    cv::resize(bitmap, resize_mask, cv::Size(bgr.cols, bgr.rows));


    for (int i = 0; i < bgr.rows; i++) {
        for (int j = 0; j < bgr.cols; j++) {
            auto *mask_p = resize_mask.ptr<uchar>(i, j);
            auto *bgr_p = bgr.ptr<uchar>(i, j);
            for (int k = 0; k < 3; k++) {
                auto value = mask_p[k]*alpha + bgr_p[k]*(1-alpha);
                mask_p[k] = value;
            }
        }
    }


    std::cout << "done!" << std::endl;

    cv::imshow("", resize_mask);
    cv::waitKey(0);

    return 0;
}
