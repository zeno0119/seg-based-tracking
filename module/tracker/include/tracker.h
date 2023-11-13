#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "inference_engine.h"
#include <dlib/optimization/max_cost_assignment.h>
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

  public:
    Tracker(const std::string& path, int width, int height) : width_(width), height_(height), sWidth_(4), engine_(InferenceEngine(path, width)),
                                                              frameno_(0), maxObjectID_(0) {
    }
    int frameno_, maxObjectID_;
    void update(const cv::Mat& img);
    cv::Mat show(const cv::Mat& img);
};

#endif