#include <iostream>
#include <cmath>
#include "MapFrames.h"

#define INF 100500

MapFrames::MapFrames(const rapidjson::Document &doc) {
    this->focal_length = doc["focal_length"].GetDouble();

    CameraInfo cam;
    switch (int(round(doc["matrix_type"].GetDouble() * 1000000.0))) {
        case 5500: {
            // KAI4050
            camera_info.matrix_width_mm = 12.85;
            camera_info.matrix_height_mm = 9.64;
            camera_info.matrix_width_pxl = 2336;
            camera_info.matrix_height_pxl = 1752;
            break;
        }
        case 5860: {
            // Imx249
            camera_info.matrix_width_mm = 11.2512;
            camera_info.matrix_height_mm = 6.328;
            camera_info.matrix_width_pxl = 1920;
            camera_info.matrix_height_pxl = 1080;
            break;
        }
        case 2900: {
            // Imx291 NO
            camera_info.matrix_width_mm = 12.85;
            camera_info.matrix_height_mm = 9.64;
            camera_info.matrix_width_pxl = 2336;
            camera_info.matrix_height_pxl = 1752;
            break;
        }
        case 3450: {
            // Imx267 Imx265 Imx252
            camera_info.matrix_width_mm = 14.13;
            camera_info.matrix_height_mm = 6.07;
            camera_info.matrix_width_pxl = 4096;
            camera_info.matrix_height_pxl = 1760;
            break;
        }
        default: {
            camera_info.matrix_width_mm = 12.85;
            camera_info.matrix_height_mm = 9.64;
            camera_info.matrix_width_pxl = 2336;
            camera_info.matrix_height_pxl = 1752;
        }
    }

    this->camera_info = cam;

    auto frames = doc["frames"].GetArray();
    std::vector<Frame> loc_frames;
    for (const auto &frame: doc["frames"].GetArray()) {
        Frame loc_frame;
        loc_frame.time = Time{frame["seconds"].GetDouble(),
                              frame["microseconds"].GetDouble()};

        for (const auto &licnum: frame["licnums"].GetArray()) {
            TRectNumber ln;
            ln.numFormat = licnum["format"].GetInt();

            std::string text = licnum["text"].GetString();
            ln.n_symbols = text.length();
            for (int i = 0; i < text.length(); ++i) {
                ln.text16[i] = (unsigned short) text[i]; // unsigned short
            }

            // TODO add certainties to parser

            auto array_x = licnum["x"].GetArray();
            auto array_y = licnum["y"].GetArray();
            for (int i = 0; i < 4; ++i) {
                ln.x[i] = array_x[i].GetUint();
                ln.y[i] = array_y[i].GetUint();
            }

            loc_frame.licnums.emplace_back(ln);
        }

        for (const auto &radar: frame["radar_targets"].GetArray()) {
            RadarTarget rt;
            rt.id = radar["id"].GetInt();
            rt.x = radar["x"].GetDouble();
            rt.y = radar["y"].GetDouble();
            rt.xspeed = radar["xspeed"].GetDouble();
            rt.yspeed = radar["yspeed"].GetDouble();
            rt.carlen = radar["carlen"].GetDouble();
            loc_frame.radar_targets.emplace_back(rt);
        }

        loc_frames.emplace_back(loc_frame);
    }
    this->frames = loc_frames;
}

double get_closest_target_speed(const coords3D c3d,
                                const std::vector<RadarTarget> &radar_targets) {
    double dist = INF;
    size_t id = -1;
    for (size_t i = 0; i < radar_targets.size(); ++i) {
        double loc_dist = std::sqrt(std::pow(c3d.x - radar_targets[i].x, 2) +
                                    std::pow(c3d.y - radar_targets[i].y, 2));
        if (loc_dist < dist) {
            dist = loc_dist;
            id = i;
        }
    }

    return std::sqrt(radar_targets[id].xspeed * radar_targets[id].xspeed +
                     radar_targets[id].yspeed * radar_targets[id].yspeed);
}

std::map<std::string, std::vector<CutFrame>>
MapFrames::fill_structure(const Calc3x4 &c) {


    std::map<std::string, std::vector<CutFrame>> map;
    for (const auto &frame: this->frames) {
        for (const auto &ln: frame.licnums) {
            double licnum_width_in_meters;
            switch (ln.numFormat) {
                case 42139648: {
                    licnum_width_in_meters = 0.445;
                }
                case 42139649: {
                    licnum_width_in_meters = 0.46;
                }
                case 42139650: {
                    licnum_width_in_meters = 0.44;
                }
                case 42139651: {
                    licnum_width_in_meters = 0.435;
                }
                case 42139652: {
                    licnum_width_in_meters = 0.45;
                }
                case 42139653: {
                    licnum_width_in_meters = 0.44;
                }
                case 42139654: {
                    licnum_width_in_meters = 0.45;
                }
                case 42139655: {
                    licnum_width_in_meters = 0.17;
                }
                case 42139656: {
                    licnum_width_in_meters = 0.405;
                }
                case 42139657: {
                    licnum_width_in_meters = 0.405;
                }
                case 42139658: {
                    licnum_width_in_meters = 0.19;
                }
                case 42139659: {
                    licnum_width_in_meters = 0.20;
                }
                case 42139660: {
                    licnum_width_in_meters = 0.42;
                }
                default: {
                    licnum_width_in_meters = 0.44;
                }
            }


            std::string s = ln.toString();
            CutFrame cf;
            cf.time = frame.time;

            coords3D c3d;
            cf.coords = c3d;

            cf.radar_speed = get_closest_target_speed(c3d, frame.radar_targets);

            if (map.find(s) == map.end()) {
                // not found
                map[s] = {cf};
            } else {
                // found
                map[s].emplace_back(cf);
            }
        }

    }

    return map;
}
