#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "inference_engine.h"
#include <dlib/optimization/max_cost_assignment.h>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>
#include <vector>

class Tracker {
  private:
    InferenceEngine engine_;
    std::vector<Object> objects_;
    int width_, height_;
    int sWidth_;
    bool isRetainAspectRate_;
    std::string outputpath_;
    std::ofstream ofs_;
    float scalex_, scaley_;

  public:
    Tracker(const std::string& path, int width, int height, const std::string& outputpath) : width_(width), height_(height), sWidth_(4), engine_(InferenceEngine(path, width, height)), frameno_(0), maxObjectID_(0), outputpath_(outputpath), ofs_(std::ofstream(outputpath)) {
    }
    int frameno_, maxObjectID_;
    size_t microseccounter;
    void update(const cv::Mat& img, const int& frameno);
    cv::Mat show(const cv::Mat& img);
    void pushLog(std::string path) {
        engine_.pushLog(path);
    }
};

#endif