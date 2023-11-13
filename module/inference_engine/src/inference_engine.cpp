#include "inference_engine.h"
using namespace cv;
// insert segment [l, r]
void SegMap::insert(signed l, signed r) {
    auto itl = upper_bound(l), itr = upper_bound(r);
    if (itl != begin()) {
        if ((--itl)->second < l)
            ++itl;
    }
    if (itl != itr) {
        l = std::min(l, itl->first);
        r = std::max(r, std::prev(itr)->second);
        erase(itl, itr);
    }
    (*this)[l] = r;
}
// remove segment [l, r]
void SegMap::remove(signed l, signed r) {
    auto itl = upper_bound(l), itr = upper_bound(r);
    if (itl != begin()) {
        if ((--itl)->second < l)
            ++itl;
    }
    if (itl == itr)
        return;
    int tl = std::min(l, itl->first), tr = std::max(r, std::prev(itr)->second);
    erase(itl, itr);
    if (tl < l)
        (*this)[tl] = l - 1;
    if (r < tr)
        (*this)[r + 1] = tr;
}

auto SegMap::getAll() const noexcept {
    return *this;
}

float calcBoxIoU(const Object& a, const Object& b) {
    if (a.xmin > b.xmax || b.xmin > a.xmax)
        return 0;
    if (a.ymin > b.ymax || b.ymin > a.ymax)
        return 0;
    auto xmin = std::max(a.xmin, b.xmin);
    auto xmax = std::min(a.xmax, b.xmax);
    auto ymin = std::max(a.ymin, b.ymin);
    auto ymax = std::min(a.ymax, b.ymax);
    float sa = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float sb = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    float sab = (xmax - xmin) * (ymax - ymin);
    return sab / (sa + sb - sab);
}

std::vector<Object> InferenceEngine::forward(const Mat& img) {
    auto t1 = std::chrono::steady_clock::now();
    float constexpr IoUth = 0.45;
    float constexpr confth = 0.25;
    CV_Assert(img.type() == CV_8UC3 || img.type() == CV_8UC1);
    cv::Mat imgr;
    cv::resize(img, imgr, Size(), (float)size_ / img.cols, (float)size_ / img.rows);
    Mat blob;
    dnn::blobFromImage(imgr, blob, 1.0 / 255.0);
    engine_.setInput(blob);
    std::vector<cv::Mat> outputs;
    auto t2 = std::chrono::steady_clock::now();
    engine_.forward(outputs, engine_.getUnconnectedOutLayersNames());
    auto t3 = std::chrono::steady_clock::now();
    auto bbs = outputs[0];
    auto seg = outputs[1];
    const auto sizes = bbs.size;
    // std::cout << sizes << std::endl;
    // std::cout << sizes[0] << ", " << sizes[1] << ", " << sizes[2] << std::endl;
    // xyxyと仮定してやる
    auto bbsp = (float*)bbs.data;
    std::vector<Object> objs;
    for (size_t i = 0; i < sizes[2]; i++) {
        for (size_t j = 4; j < 36; j++) {
            auto cls = bbsp[j * sizes[2] + i];
            if (cls < confth)
                continue;
            else {
                auto xc = bbsp[0 * sizes[2] + i];
                auto yc = bbsp[1 * sizes[2] + i];
                auto w = bbsp[2 * sizes[2] + i];
                auto h = bbsp[3 * sizes[2] + i];
                objs.push_back(Object(xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2, cls, j - 4));
                break;
            }
        }
    }
    auto t4 = std::chrono::steady_clock::now();
    // std::cout << objs.size() << std::endl;
    // std::cout << "nms" << std::endl;
    std::sort(objs.begin(), objs.end(), [](const Object& a, const Object& b) { return a.conf > b.conf; });
    for (size_t i = 0; i < objs.size(); i++) {
        // std::cout << i << std::endl;
        for (auto jtr = objs.begin() + i + 1; jtr != objs.end();) {
            // std::cout << calcBoxIoU(objs[i], *jtr) << ", ";
            if (calcBoxIoU(objs[i], *jtr) > IoUth) {
                jtr = objs.erase(jtr);
            } else {
                jtr++;
            }
        }
        // std::cout << std::endl;
    }
    auto t5 = std::chrono::steady_clock::now();
    auto segsize = seg.size;
    // create segmap
    // std::cout << seg.size << std::endl;
    float constexpr segconf = 0.5;
    bool first = true;
    auto segp = (float*)seg.data;
    for (auto& el : objs) {
        std::vector<SegMap> s;
        for (size_t x = el.xmin / 4; x < el.xmax / 4; x++) {
            auto tmp = SegMap();
            auto tmpidx = el.ymin / 4;
            bool mode = false;
            for (size_t y = el.ymin / 4; y < el.ymax / 4; y++) {
                if (segp[6 * segsize[3] * segsize[2] + y * segsize[3] + x] > segconf) {
                    if (mode == false) {
                        mode = true;
                        tmpidx = y;
                    }
                } else if (mode == true) {
                    mode = false;
                    tmp.insert(tmpidx * 4, (y - 1) * 4);
                }
            }
            s.push_back(tmp);
        }
        el.setSeg(s);
    }
    auto t6 = std::chrono::steady_clock::now();
    // auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    // auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    // auto duration34 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    // auto duration45 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    // auto duration56 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
    // printf("pre: %lf, inf: %lf, box: %lf, nms: %lf, segm: %lf\n", duration12 / 1000.0, duration23 / 1000.0, duration34 / 1000.0, duration45 / 1000.0, duration56 / 1000.0);

    return objs;
}
