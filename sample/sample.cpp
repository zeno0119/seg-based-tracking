#include "tracker.h"
#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
template <typename... Args>
std::string format(const std::string& fmt, Args... args) {
    size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args...);
    std::vector<char> buf(len + 1);
    std::snprintf(&buf[0], len + 1, fmt.c_str(), args...);
    return std::string(&buf[0], &buf[0] + len);
}
int main() {
    std::cout << "Start tracking" << std::endl;
    std::string path = "/home/zeno0119/Desktop/yolo/yolov8n-seg.onnx";
    std::string p = "/movie/MOT17/train/MOT17-11/img/%06d.jpg";
    Tracker t(path, 320, 320);
    for (int i = 1;; i++) {
        auto l = format(p, i);
        std::cout << std::setw(100) << l << std::endl;
        auto img = cv::imread(l);
        auto t1 = std::chrono::steady_clock::now();

        if (img.empty()) {
            std::cerr << "Image Read Error" << std::endl;
            break;
        }
        t.update(img);
        auto t2 = std::chrono::steady_clock::now();
        auto du = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        printf("duration: %lf ms.", (double)du / 1000.0);
        auto res = t.show(img);
        cv::imshow("result", res);
        printf("\r\e[3A");
        cv::waitKey(1);
    }
    return 0;
}