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

auto OR(const SegMap& a, const SegMap& b) {
    auto res = a;
    for (const auto& el : a.getAll()) {
        res.insert(el.first, el.second);
    }
    return res;
}

auto SegMap::NOT() const noexcept {
    SegMap res;
    res.insert(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    for (const auto& el : *this) {
        res.remove(el.first, el.second);
    }
    return res;
}

auto AND(const SegMap& a, const SegMap& b) noexcept {
    auto res = a;

    for (const auto& el : b.NOT().getAll()) {
        res.remove(el.first, el.second);
    }
    return res;
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

float calcSegIoU(const Object& a, const Object& b) {
    if (a.xmin > b.xmax || b.xmin > a.xmax)
        return 0;
    if (a.ymin > b.ymax || b.ymin > a.ymax)
        return 0;

    float sa = 0, sb = 0, sab = 0;
    for (const auto& el : a.seg) {
        for (const auto& e : el.getAll()) {
            sa += 4 * (e.second - e.first);
        }
    }
    for (const auto& el : b.seg) {
        for (const auto& e : el.getAll()) {
            sb += 4 * (e.second - e.first);
        }
    }

    int start = std::max(a.xmin, b.xmin) / 4;
    int end = std::min(a.xmax, b.xmax) / 4;
    for (size_t i = start; i < end; i++) {
        for (const auto& el : AND(a.seg[i - a.xmin / 4], b.seg[i - b.xmin / 4]).getAll()) {
            sab += 4 * (el.second - el.first);
        }
    }
    return sab / (sa + sb - sab);
}

std::vector<Object> InferenceEngine::forward(const Mat& img) {
    auto t1 = std::chrono::steady_clock::now();
    float constexpr IoUth = 0.45;
    float constexpr confth = 0.15;
    CV_Assert(img.type() == CV_8UC3 || img.type() == CV_8UC1);
    Mat imgr;
    resize(img, imgr, Size(), (float)width_ / img.cols, (float)height_ / img.rows);
    if (retainAspectRate_) {
        auto padl = std::max(imgr.cols, imgr.rows) - imgr.cols;
        auto padb = std::max(imgr.cols, imgr.rows) - imgr.rows;
        copyMakeBorder(imgr, imgr, 0, padb, padl, 0, BORDER_CONSTANT, Scalar::all(0));
    }
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
    float max_conf = 0;
    float max_idx = -1;
    for (size_t i = 0; i < sizes[2]; i++) {
        for (size_t j = 4; j < 36; j++) {
            auto cls = bbsp[j * sizes[2] + i];
            if (cls > max_conf) {
                max_conf = cls;
                max_idx = j;
            }
        }

        if (max_conf > confth && max_idx != -1) {
            auto xc = bbsp[0 * sizes[2] + i];
            auto yc = bbsp[1 * sizes[2] + i];
            auto w = bbsp[2 * sizes[2] + i];
            auto h = bbsp[3 * sizes[2] + i];
            objs.push_back(Object(xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2, max_conf, max_idx - 4));
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
        if (el.cls != 0)
            continue;
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
    auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    auto duration36 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t3).count();
    pre.push_back(duration12);
    inf.push_back(duration23);
    post.push_back(duration36);
    // printf("pre: %lf, inf: %lf, box: %lf, nms: %lf, segm: %lf\n", duration12 / 1000.0, duration23 / 1000.0, duration34 / 1000.0, duration45 / 1000.0, duration56 / 1000.0);

    return objs;
}

void InferenceEngine::pushLog(std::string path) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "file not found" << std::endl;
        return;
    }
    for (size_t i = 0; i < pre.size(); i++) {
        ofs << pre[i] << " " << inf[i] << " " << post[i] << std::endl;
    }
}