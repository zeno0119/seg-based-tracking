#ifndef __INFERENCE_ENGINE_H__
#define __INFERENCE_ENGINE_H__
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <vector>

// Description: 区間をsetで管理するデータ構造(なお実装はmap)．各クエリO(log区間数)．

// #### attention! : [l, r] ( include r, not [l, r) )
struct SegMap : public std::map<signed, signed> {
  public:
    SegMap() {}
    // insert segment [l, r]
    void insert(signed l, signed r);
    // remove segment [l, r]
    void remove(signed l, signed r);
    auto getAll() const noexcept;
    auto NOT() const noexcept;
};

struct Object {
    int undetCounter, ObjectID, matchCounter;
    int cls;
    int xmin, xmax, ymin, ymax;
    float conf;
    std::vector<SegMap> seg;
    Object(float xmi, float xma, float ymi, float yma, float c, int cls_) : undetCounter(0), ObjectID(-1), matchCounter(0),
                                                                            xmin(xmi), xmax(xma), ymin(ymi), ymax(yma), conf(c), cls(cls_) {
    }
    void setSeg(std::vector<SegMap> seg_) {
        seg = seg_;
    }
};

float calcBoxIoU(const Object& a, const Object& b);

float calcSegIoU(const Object& a, const Object& b);

class InferenceEngine {
  private:
    decltype(cv::dnn::readNetFromONNX("path")) engine_;
    int width_, height_;
    int sWidth_;
    bool retainAspectRate_;
    std::vector<int64_t> pre, inf, post;

  public:
    InferenceEngine(const std::string& path, int width, int height) : width_(width), height_(height), sWidth_(4), retainAspectRate_(width != height) {
        engine_ = cv::dnn::readNetFromONNX(path);
        engine_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        engine_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    std::vector<Object> forward(const cv::Mat& img);
    void pushLog(std::string ofs);
};

#endif