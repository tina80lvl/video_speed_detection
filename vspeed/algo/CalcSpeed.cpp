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
            ans[key].emplace_back(std::make_pair(vspeed, val[j].radar_speed));
        }
    }
    return ans;
}

std::pair<double, double> speed_detection_quality(const SPD &speeds_by_licnum) {
    double all_total = 0;
    double all_frameby = 0;
    for (auto[key, val]: speeds_by_licnum) {
        all_frameby += val.size();
        double ln_total = 0;
        for (const auto &p: val) {
            ln_total += std::abs(p.first - p.second);
        }
        double ln_avg = ln_total / val.size();
        all_total += ln_avg;
    }

    return {all_total / speeds_by_licnum.size(),
            all_frameby / speeds_by_licnum.size()};
}
