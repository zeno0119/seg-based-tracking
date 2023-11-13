#include "tracker.h"
using namespace cv;

void Tracker::update(const Mat& img) {
    std::cout << std::setw(100) << "frame no: " << frameno_++ << std::endl;
    float constexpr IoUth = 0.1;
    auto objs = engine_.forward(img);
    if (frameno_ == 1) {
        objects_ = objs;
        return;
    }
    std::vector<Object> undetObjects, newObjects, matchedObjects, prevMatch, newMatch;
    // matching
    for (auto& itr : objs) {
        auto del = true;
        for (auto& jtr : objects_) {
            if (calcBoxIoU(itr, jtr) > IoUth) {
                del = false;
                break;
            }
        }
        if (del) {
            newObjects.push_back(itr);
        } else {
            newMatch.push_back(itr);
        }
    }

    for (auto& itr : objects_) {
        auto del = true;
        for (auto& jtr : objs) {
            if (calcBoxIoU(itr, jtr) > IoUth) {
                del = false;
                break;
            }
        }
        if (del) {
            undetObjects.push_back(itr);
        } else {
            prevMatch.push_back(itr);
        }
    }

    auto msize = std::max(prevMatch.size(), newMatch.size());
    dlib::matrix<int> cost(msize, msize);
    for (int i = 0; i < msize; i++) {
        for (int j = 0; j < msize; j++) {
            cost(i, j) = 0;
            if (i >= prevMatch.size())
                break;
            if (j >= newMatch.size())
                break;

            cost(i, j) = (int)(10000 * calcBoxIoU(prevMatch[i], newMatch[j]));
        }
    }
    int d = 0, n = 0, u = 0;
    auto assignment = dlib::max_cost_assignment(cost);
    std::vector<Object> res;
    for (int i = 0; i < assignment.size(); i++) {
        auto j = assignment[i];
        if (i < prevMatch.size() && j < newMatch.size()) {
            // matched
            u += 1;
            newMatch[j].undetCounter = 0;
            newMatch[j].ObjectID = prevMatch[i].ObjectID;
            newMatch[j].matchCounter = prevMatch[i].matchCounter + 1;
            if (newMatch[j].matchCounter >= 3 && newMatch[j].ObjectID == -1) {
                newMatch[j].ObjectID = maxObjectID_++;
            }

            res.push_back(newMatch[j]);
        } else if (i >= prevMatch.size()) {
            // new object
            n += 1;
            res.push_back(newMatch[j]);
        } else if (j >= newMatch.size()) {
            // undetected
            d += 1;
            prevMatch[i].undetCounter += 1;
            if (prevMatch[i].undetCounter < 2) {
                res.push_back(prevMatch[i]);
            }
        }
    }
    std::cout << std::setw(100) << "undetected: " << d << ", new: " << n << ", matched: " << u << std::endl;

    objects_ = res;
}

Mat Tracker::show(const cv::Mat& img) {
    Mat imgr;
    resize(img, imgr, cv::Size(), (float)width_ / img.cols, (float)width_ / img.rows);
    auto mask = Mat(imgr.size(), img.type(), Scalar::all(0));
    for (auto& el : objects_) {
        if (el.undetCounter != 0 || el.ObjectID == -1)
            continue;
        auto tl = Point2d((int)el.xmin, (int)el.ymin);
        auto br = Point2d((int)el.xmax, (int)el.ymax);
        auto rect = Rect((int)el.xmin, (int)el.ymin, (int)(el.xmax - el.xmin), (int)(el.ymax - el.ymin));
        rectangle(mask, rect, Scalar::all(255));
        auto id = std::to_string(el.ObjectID);
        putText(mask, id, tl, 1, 0.5, Scalar::all(255));
    }
    Mat res;
    addWeighted(imgr, 0.5, mask, 0.5, 0.0, res);
    return res;
}