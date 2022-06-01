#include <iostream>
#include <cmath>
#include "CalcSpeed.h"

double dist_in_3d(const coords3D &p1, const coords3D &p2) {
    return std::sqrt(
            (p1.x - p2.x) * (p1.x - p2.x) +
            (p1.y - p2.y) * (p1.y - p2.y) +
            (p1.zl - p2.zl) * (p1.zl - p2.zl)
    );
}

SPD calculate_speeds(int frame_dist,
                     std::map<std::string, std::vector<CutFrame>> &map) {
    SPD ans;
    for (auto[key, val]: map) {
        if (val.size() <= frame_dist) continue;
        for (size_t i = 0; i < val.size() - frame_dist; ++i) {
            size_t j = i + frame_dist;

            double dist = dist_in_3d(val[i].coords, val[j].coords);
            double vspeed = dist / (val[j].time - val[i].time).diff();
//            std::cerr << "calc speed: " << vspeed << ", real: " << val[j].radar_speed << std::endl;
            ans[key].emplace_back(std::make_pair(vspeed, val[j].radar_speed));
        }
    }
    return ans;
}

std::pair<double, double> speed_detection_quality(const SPD &speeds_by_licnum) {
    double all_total = 0;
    double all_frameby = 0;
    double cnt = 0, all_cnt = 0;
    double mins = 1000;
    double maxs = -1;
    for (auto[key, val]: speeds_by_licnum) {
        all_frameby += val.size();
        double ln_total = 0;
        cnt = 0;
        for (const auto &p: val) {
            if (p.first == -1) continue;
            ln_total += std::abs(p.first - p.second);
            ++cnt;
            mins = std::min(mins, p.second);
            maxs = std::max(maxs, p.second);
        }
        double ln_avg = ln_total / cnt;
        all_total += ln_avg;
        all_cnt += cnt;
    }

//    std::cerr << "min " << mins << " max " << maxs << std::endl;

    return {all_total / all_cnt,
            all_frameby / speeds_by_licnum.size()};
}

double false_positive(const SPD &speeds_by_licnum, double limit) {
    double sum = 0;
    double cnt = 0;
    for (auto[key, val]: speeds_by_licnum) {
        for (const auto &p: val) {
            if (p.first == -1) continue;

//            std::cerr << p.second << ", ";

//            if (std::abs(p.first - p.second) > limit)
//                std::cerr << p.first << ' ' << p.second << std::endl;
            sum += std::abs(p.first - p.second) > limit;
            ++cnt;
        }
    }

    return sum / cnt;
}

int missed(const SPD &speeds_by_licnum) {
    int cnt = 0;
    for (auto[key, val]: speeds_by_licnum) {
        int loc = 0;
        for (const auto &p: val) {

            if (p.first == -1 && p.second > 0) {
                ++loc;
            }
        }
//        std::cerr << "missed " << loc << " out of " << val.size() << std::endl;
        if (loc == val.size()) {
            ++cnt;
        }
    }

    return cnt;
}