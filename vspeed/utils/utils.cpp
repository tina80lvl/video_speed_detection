#include <iostream>

#include "../libs/Calc.h"

#include "../algo/MapFrames.h"
#include "../algo/CalcSpeed.h"

#include "utils.h"
#include "json.h"


void save_coords(std::ofstream &save_file,
                 const std::map<std::string, std::vector<CutFrame>> &map) {

    for (auto[key, val]: map) {
        for (const auto &p: val) {
            save_file << key << ',' << std::to_string(p.time.sec) << ','
                      << std::to_string(p.time.usec) << ','
                      << std::to_string(p.coords.x) << ','
                      << std::to_string(p.coords.y) << ','
                      << std::to_string(p.coords.zl) << ','
                      << std::to_string(p.radar_speed) << '\n';
        }
    }
}

void save_all_speeds(std::ofstream &save_file, SPD &speeds, int frame_dist) {

    for (auto[key, val]: speeds) {
        for (const auto &p: val) {
            save_file << std::to_string(frame_dist) << ',' << key << ','
                      << std::to_string(p.first) << ','
                      << std::to_string(p.second) << '\n';
        }
    }
}

std::string process_file(const std::string &path, std::ofstream &speeds_file,
                         std::ofstream &coords_file) {
    std::cout << "Processing file: " << path << std::endl;
    rapidjson::Document doc = JsonParser::getJsonDocument(path);

    auto arr = doc["matrix3x4"].GetArray();
    double m3x4_local[12];
    for (int i = 0; i < 12; ++i) {
        m3x4_local[i] = arr[i].GetDouble();
    }

    Calc3x4 c;
    c.setMatrix3x4(m3x4_local);

    MapFrames map_frames{doc};

    std::map<std::string, std::vector<CutFrame>> map = map_frames.fill_structure(
            c);

    // Export 3D-calculated coordinates
    save_coords(speeds_file, map);

    std::string file_name = path.substr(path.find_last_of("/\\") + 1);
    std::string report;
    for (int i = 2; i < 3; ++i) {
        auto speeds = calculate_speeds(i, map);

        // Export calculated speeds
        save_all_speeds(speeds_file, speeds, i);

        const auto &mistake = speed_detection_quality(speeds);
        const auto &false_pos = false_positive(speeds, 19);
        const auto &missed_targets = missed(speeds);

        report += file_name + ',' +
                  std::to_string(map.size()) + ',' +
                  std::to_string(mistake.second) + ',' +
                  std::to_string(mistake.first) + ',' +
                  std::to_string(mistake.first * 3.6) + ',' +
                  std::to_string(false_pos) + '\n';
    }

    return report;
}
