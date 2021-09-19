
#include <net.h>

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

#define target_size 640
#define prob_threshold 0.5f
#define nms_threshold 0.45f
#define pad_ratio 0.5f


struct FaceObject
{
    cv::Rect_<float> rect;
    float prob;
};


static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p) i++;
        while (faceobjects[j].prob < p) j--;
        if (i <= j) std::swap(faceobjects[i++], faceobjects[j--]);
    }

    {
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }

        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}


static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}


static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales) {
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = 0;
    const float cy = 0;

    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, std::vector<FaceObject>& faceobjects) {
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++) {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++) {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold) {
                    // insightface/detection/scrfd/mmdet/models/dense_heads/scrfd_head.py _get_bboxes_single()
                    float dx = bbox.channel(0)[index] * feat_stride;
                    float dy = bbox.channel(1)[index] * feat_stride;
                    float dw = bbox.channel(2)[index] * feat_stride;
                    float dh = bbox.channel(3)[index] * feat_stride;

                    // insightface/detection/scrfd/mmdet/core/bbox/transforms.py distance2bbox()
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.prob = prob;

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = faceobjects[i].rect.area();

    for (int i = 0; i < n; i++) {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const FaceObject& b = faceobjects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) keep = 0;
        }

        if (keep) picked.push_back(i);
    }
}


void detect(const cv::Mat& bgr, const ncnn::Net& net, std::vector<FaceObject>& faceobjects) {
    int w = bgr.cols;
    int h = bgr.rows;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, w, h);
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/128.f, 1/128.f, 1/128.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", in_pad);

    std::vector<FaceObject> faceproposals;

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_8", score_blob);
        ex.extract("bbox_8", bbox_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_16", score_blob);
        ex.extract("bbox_16", bbox_blob);

        const int base_size = 64;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_32", score_blob);
        ex.extract("bbox_32", bbox_blob);

        const int base_size = 256;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    qsort_descent_inplace(faceproposals);
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked);

    int face_count = picked.size();

    faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++) {
        faceobjects[i] = faceproposals[picked[i]];
        // adjust offset to original unpadded
        float x0 = (faceobjects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (faceobjects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)bgr.cols - 1), 0.f);
        y0 = std::max(std::min(y0, (float)bgr.rows - 1), 0.f);
        x1 = std::max(std::min(x1, (float)bgr.cols - 1), 0.f);
        y1 = std::max(std::min(y1, (float)bgr.rows - 1), 0.f);
        // x0, y0, w, h
        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;
    }
}

void landmark(cv::Mat &bgr, const ncnn::Net &net, const FaceObject &obj, std::vector<cv::Point2f> &landmarks)
{
    int pad = obj.rect.height;
    cv::Rect box;

    box.x = (obj.rect.x+obj.rect.width/2) - pad*pad_ratio;
    box.y = obj.rect.y;
    box.width = obj.rect.height;
    box.height = obj.rect.height;

    box.x = std::max(0.f,(float)box.x);
    box.y = std::max(0.f,(float)box.y);
    box.width = box.x + box.width<bgr.cols?box.width:bgr.cols-box.x-1;
    box.height = box.y + box.height<bgr.rows?box.height:bgr.rows-box.y-1;

    cv::Mat faceRoiImage = bgr(box).clone();

    ncnn::Extractor ex_face = net.create_extractor();
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(faceRoiImage.data, ncnn::Mat::PIXEL_BGR2RGB, faceRoiImage.cols, faceRoiImage.rows,192,192);
    const float means[3] = { 127.5f, 127.5f,  127.5f };
    const float norms[3] = { 1/127.5f, 1 / 127.5f, 1 / 127.5f };
    ncnn_in.substract_mean_normalize(means, norms);
    ex_face.input("input.1",ncnn_in);
    ncnn::Mat ncnn_out;
    ex_face.extract("482",ncnn_out);
    float *scoredata = (float*)ncnn_out.data;
    for(int i = 0; i < 468; i++) {
        cv::Point2f pt;
        pt.x = scoredata[i*3]*box.width/192 + box.x;
        pt.y = scoredata[i*3+1]*box.width/192 + box.y;
        landmarks.push_back(pt);
    }
}


void split(const std::string& str, std::vector<std::vector<int>>& contours) {  
	char* strs = new char[str.length() + 1]; 
	strcpy(strs, str.c_str());   
 
	char* d = new char[2];
    const char* dec = " ";
	strcpy(d, dec);  
 
	char *p = strtok(strs, d);
    std::vector<int> tmp;
	while(p) {  
        tmp.push_back(atoi(p));
        p = strtok(NULL, d);
	}  
    contours.push_back(tmp);
} 


int main(int argc, char** argv) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s [imagepath] [weightpath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::cout << ">>> load image done!" << std::endl;
    const char* weight_root = argv[2];  // ../../weight
    ncnn::Net face_detector, face_mesher;
    face_detector.load_param((std::string(weight_root) + "/scrfd.param").c_str());
    face_detector.load_model((std::string(weight_root) + "/scrfd.bin").c_str());
    face_mesher.load_param((std::string(weight_root) + "/facemesh.param").c_str());
    face_mesher.load_model((std::string(weight_root) + "/facemesh.bin").c_str());

    std::cout << ">>> load model weight done!" << std::endl;

    std::vector<FaceObject> faceobjects;
    detect(bgr, face_detector, faceobjects);

    std::cout << ">>> face detect done, found " << faceobjects.size() << " faces!" << std::endl;

    std::ifstream ifs(std::string(weight_root) + "/landmark_contours.txt");
    std::vector<std::vector<int>> contours;
    std::string str;
    while (getline(ifs, str)) {
        split(str, contours);
    }

    for (size_t i = 0; i < faceobjects.size(); i++) {
        std::vector<cv::Point2f> pts;
        const FaceObject& obj = faceobjects[i];

        landmark(bgr, face_mesher, obj, pts);
        std::cout << ">>> face mesh done!" << std::endl;
        // draw
        for(int i = 0; i < pts.size(); i++) cv::circle(bgr, pts[i], 1, cv::Scalar(0, 0, 255), -1);
        for (const auto& contour : contours) {
            for (int i = 0; i < contour.size() - 1; i++) {
                cv::line(bgr, pts[contour[i]], pts[contour[i + 1]], cv::Scalar(255, 0, 0), 2);
            }
        }
        cv::rectangle(bgr, obj.rect, cv::Scalar(0, 255, 0));

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > bgr.cols) x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        std::cout << ">>> face draw donw!" << std::endl;
    }

    cv::imshow("", bgr);
    cv::waitKey(0);

    return 1;
}

