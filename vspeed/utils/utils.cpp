#include <iostream>

#include "../libs/Calc.h"

#include "../algo/MapFrames.h"
#include "../algo/CalcSpeed.h"
#include "../algo/Regression.h"

#include "utils.h"
#include "json.h"

void linear_tune(std::map<std::string, std::vector<CutFrame>>& map) {
    for(auto& [key, val] : map) {
        Regression model_x{val, 'x'};
        Regression model_y{val, 'y'};
        Regression model_z{val, 'z'};

        for (size_t i = 0; i < val.size(); ++i) {
            val[i].coords.x = model_x.predict(val[i].time.get_float());
            val[i].coords.y = model_y.predict(val[i].time.get_float());
            val[i].coords.zl = model_z.predict(val[i].time.get_float());
        }
    }
}


std::string process_file(const std::string &path) {
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

//    linear_tune(map);

    std::string file_name = path.substr(path.find_last_of("/\\") + 1);
    std::string report;
    for (int i = 2; i < 3; ++i) {
        auto speeds = calculate_speeds(i, map);

        const auto &mistake = speed_detection_quality(speeds);
        const auto &false_pos = false_positive(speeds, 19);
        const auto &missed_targets = missed(speeds);

        report += file_name + ',' +
//                  std::to_string(i) + ',' +
                  std::to_string(map.size()) + ',' +
                  std::to_string(mistake.second) + ',' +
                  std::to_string(mistake.first) + ',' +
                  std::to_string(mistake.first * 3.6) + ',' +
                  std::to_string(false_pos) + '\n';
//                  std::to_string(missed_targets) + '\n';

    }

    return report;
}
