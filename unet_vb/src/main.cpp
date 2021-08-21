#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>

#define input_size 256
#define alpha 0.5
#define th 0.3


static inline void get_coords(const int &x, std::vector<int> &data) {
    int _x = 0;
    while (_x < x) {
        data.push_back(_x);
        _x += input_size;
    }
}


void crop(const cv::Mat& bgr, std::vector<cv::Mat>& inputs, std::vector<std::vector<int>> &x0y0_coords, 
    std::vector<int> &padded_size, std::vector<int> &pad01) {
    const int raw_h = bgr.rows;
    const int raw_w = bgr.cols;

    int pad0 = (input_size - raw_h % input_size) % input_size;
    int pad1 = (input_size - raw_w % input_size) % input_size;
    pad01 = {pad1, pad0};  // x, y

    cv::Mat padded;
    cv::copyMakeBorder(bgr, padded, pad0/2, pad0/2, pad1/2, pad1/2, cv::BORDER_CONSTANT, cv::Scalar(0));
    std::cout << "before padding:  h=" << raw_h << ", w=" << raw_w << std::endl;
    std::cout << "afert  padding : h=" << padded.rows << ", w=" << padded.cols << std::endl; 

    padded_size = { padded.cols, padded.rows };  // x, y

    std::vector<int> coord_ys;
    std::vector<int> coord_xs;
    get_coords(padded.rows, coord_ys);
    get_coords(padded.cols, coord_xs);

    x0y0_coords.resize(coord_xs.size() * coord_ys.size(), std::vector<int>(2, 0));
    for (int i = 0; i < coord_xs.size(); i++) {
      for (int j = 0; j < coord_ys.size(); j++) {
          x0y0_coords[i*coord_ys.size() + j] = { coord_xs[i], coord_ys[j] };

          inputs.push_back(padded(cv::Rect(coord_xs[i], coord_ys[j], input_size, input_size)).clone());
      }
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
    std::string const& path_param = std::string(weight_root) + "/unet_vb.param";
    std::string const& path_bin = std::string(weight_root) + "/unet_vb.bin";

    std::vector<cv::Mat> inputs;
    std::vector<std::vector<int>> x0y0_coords;
    std::vector<int> padded_size;  // w, h
    std::vector<int> pad01;  // x, y
    crop(bgr, inputs, x0y0_coords, padded_size, pad01);

    std::cout << ">>> forward..." << std::endl;

    ncnn::Net unet;
    unet.opt.use_vulkan_compute = true;
    unet.load_param(path_param.c_str());
    unet.load_model(path_bin.c_str());

    ncnn::Layer* sigmoid = ncnn::create_layer("Sigmoid");
    ncnn::ParamDict pd;
    ncnn::Option opt;
    sigmoid->load_param(pd);
    sigmoid->create_pipeline(opt);

    cv::Mat mask = cv::Mat::zeros(padded_size[1], padded_size[0], CV_8UC1);

    for (int i = 0; i < inputs.size(); i++) {
        cv::Mat input = inputs[i];
        const auto &coord = x0y0_coords[i];

        // ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_BGR2RGB, 
        //         input.cols, input.rows, input_size, input_size);
        ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_BGR2RGB, input_size, input_size);

        const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
        const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);  // c, h, w

        ncnn::Extractor ex = unet.create_extractor();
        ex.input("input", in);
        ncnn::Mat logits;
        ex.extract("logits", logits);  // 1, h, w 

        sigmoid->forward_inplace(logits, opt);

        cv::Mat cur_mask = cv::Mat::zeros(logits.h, logits.w, CV_8UC1);
        float *logits_p = (float*)logits.data;
        auto *mask_p = cur_mask.ptr<uchar>();

        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                if (logits_p[i*input_size + j] > th) {
                    mask_p[i*input_size + j] = cv::saturate_cast<uchar>(255);
                }
            }
        }

        cv::Rect rect = cv::Rect((int)coord[0], (int)coord[1], input_size, input_size);
        cv::Mat dstMask = mask(rect);
        cur_mask.colRange(0, cur_mask.cols).copyTo(dstMask);
    }
    // crop padding
    cv::Rect rect = cv::Rect(pad01[0]/2, pad01[1]/2, bgr.cols, bgr.rows);
    cv::Mat visMask = mask(rect);
    std::cout << ">>> crop mask ==> h:" << visMask.rows << "  w: " << visMask.cols << std::endl;

    cv::Mat labelmask;
    int num_of_labels = connectedComponents(visMask, labelmask);
    std::cout << ">>> found " << (num_of_labels - 1) << " vascular bundle!\n";

    std::vector<cv::Vec3b> colors(num_of_labels);
    for (int i = 1; i < num_of_labels; i++) {
        colors[i] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    for (int i = 0; i < bgr.rows; i++) {
        for (int j = 0; j < bgr.cols; j++) {
            int label = (int)labelmask.at<int>(i, j);
            if (label != 0) {
                bgr.at<cv::Vec3b>(i, j) = colors[label]*alpha + bgr.at<cv::Vec3b>(i, j)*(1-alpha);
            }
        }
    }

    cv::imshow("", bgr);
    cv::waitKey(0);

    std::cout << "done" << std::endl;

    return 0;
}
