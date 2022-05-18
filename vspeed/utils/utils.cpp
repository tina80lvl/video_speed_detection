#include <iostream>

#include "../libs/Calc.h"

#include "../algo/MapFrames.h"
#include "../algo/CalcSpeed.h"

#include "utils.h"
#include "json.h"

std::string process_file(const std::string &path) {
    std::cerr << "Processing file: " << path << std::endl;
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

    std::string file_name = path.substr(path.find_last_of("/\\") + 1);
    std::string report;
    for (int i = 40; i < 80; ++i) {
        auto speeds = calculate_speeds(i, map);

        const auto &mistake = speed_detection_quality(speeds);

        report += file_name + ',' + std::to_string(i) + ',' +
                  std::to_string(map.size()) + ',' +
                  std::to_string(mistake.second) + ',' +
                  std::to_string(mistake.first) + ',' +
                  std::to_string(mistake.first * 3.6) + '\n';

    }
    return report;
}
