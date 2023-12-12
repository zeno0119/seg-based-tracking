#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <vector>

struct obj {
  public:
    float xmin, ymin;
    float width, height;
    int objectID;
    int frameno;
    obj(float xmin_, float ymin_, float width_, float height_, int objectID_, int frameno_) : xmin(xmin_), ymin(ymin_), width(width_), height(height_), objectID(objectID_), frameno(frameno_) {}
};

template <typename... Args>
std::string format(const std::string& fmt, Args... args) {
    size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args...);
    std::vector<char> buf(len + 1);
    std::snprintf(&buf[0], len + 1, fmt.c_str(), args...);
    return std::string(&buf[0], &buf[0] + len);
}

using namespace cv;

int main() {
    std::string path = "/movie/MOT17/train/MOT17-02/img/%06d.jpg";
    std::ifstream ifs("/movie/yoloResult/aspx320/MOT17-02.txt");

    std::string tmp;

    std::vector<std::vector<obj>> objects;
    std::vector<obj> tmpv;
    int frameno = 1;
    while (std::getline(ifs, tmp)) {
        std::stringstream ss(tmp);
        std::vector<std::string> r;
        std::string t;
        while (std::getline(ss, t, ' ')) {
            r.push_back(t);
        }
        if (frameno != std::stoi(r[0])) {
            frameno = std::stoi(r[0]);
            objects.push_back(tmpv);
            tmpv = std::vector<obj>();
        }
        tmpv.push_back(obj(std::stof(r[2]), std::stof(r[3]), std::stof(r[4]), std::stof(r[5]), std::stoi(r[1]), std::stoi(r[0])));
    }
    for (int i = 1;; i++) {
        auto l = format(path, i);
        auto img = cv::imread(l);
        if (img.empty()) {
            std::cerr << "img read error" << std::endl;
            break;
        }

        for (const auto& el : objects[i - 1]) {
            const auto tl = Point2d((int)el.xmin, (int)el.ymin);
            const auto br = Point2d((int)(el.xmin + el.width), (int)(el.ymin + el.height));
            const auto rect = Rect(tl, br);
            rectangle(img, rect, Scalar::all(255));
        }
        imshow("result", img);
        auto k = waitKey(1);
        if (k == 'q') {
            break;
        }
    }
    return 0;
}