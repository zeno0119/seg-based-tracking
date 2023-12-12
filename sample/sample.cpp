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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Invalid argment" << std::endl;
        std::cerr << "Usage: sample <resulution> <sequence>" << std::endl;
    }
    auto resol = std::stoi(argv[1]);
    auto sequence = argv[2];
    std::cout << "Start tracking" << std::endl;
    std::string path = "/home/zeno0119/Desktop/yolo/yolov8n-seg%3d.onnx";
    std::string p = "/movie/MOT17/train/%s/img/%06d.jpg";
    std::string outpath = "/movie/yoloResult/aspx%3d/%s/%06d.jpg";
    std::string outputpath = "/movie/yoloResult/aspx%3d/%s.txt";
    auto nnpath = format(path, resol);
    std::cout << "read net from: " << nnpath << std::endl;
    std::vector<int64_t> v_duration;
    auto resoly = (float)resol / 16.0 * 9.0;
    Tracker t(nnpath, resol, resoly, format(outputpath, resol, sequence));
    for (int i = 1;; i++) {
        auto l = format(p, sequence, i);
        auto outl = format(outpath, resol, sequence, i);
        // std::cout << std::left << l << std::endl;
        auto img = cv::imread(l);
        auto t1 = std::chrono::steady_clock::now();

        if (img.empty()) {
            std::cerr << "Image Read Error" << std::endl;
            break;
        }
        t.update(img, i);
        auto t2 = std::chrono::steady_clock::now();
        auto du = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        v_duration.push_back(du);
        // printf("duration: %lf ms.\n", (double)du / 1000.0);
        auto res = t.show(img);
        cv::imwrite(outl, res);
        // cv::imshow("result", res);
        // printf("\r\e[4A");
        // cv::waitKey(1);
    }
    std::sort(v_duration.begin(), v_duration.end());
    if (v_duration.size() % 2 == 0) {
        const auto s = v_duration.size();
        std::cout << "mean of duration: " << (float)((v_duration[s / 2] + v_duration[s / 2 - 1]) / 2) / 1000 << "ms" << std::endl;
    } else {
        std::cout << "mean of duration: " << (float)(v_duration[v_duration.size() / 2]) / 1000 << "ms" << std::endl;
    }
    auto logpath = "/home/zeno0119/Desktop/yolo/logx480.dat";
    t.pushLog(logpath);
    return 0;
}