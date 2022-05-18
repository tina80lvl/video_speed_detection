#ifndef VSPEED_MAPFRAMES_H
#define VSPEED_MAPFRAMES_H

#include <vector>
#include <map>
#include "rapidjson/document.h"
#include "../libs/Types.h"
#include "../libs/Calc.h"

struct RadarTarget {
    int id;
    double x;
    double y;

    double xspeed;
    double yspeed;

    double carlen;
};

struct Car {
    TRectNumber licnum;
    double speed;
};

struct Time {
    double sec;
    double usec;

    Time operator-(const Time &a) const {
        return Time{sec - a.sec, usec - a.usec};
    }

    double diff() {
        return (double) sec + (double) usec / 1000000.0;
    }
};

struct Frame {
    Time time;
    std::vector<TRectNumber> licnums;
    std::vector<RadarTarget> radar_targets;
};

struct coords3D {
    double x = 0;
    double y = 0;
    double zl = 0;
    double zr = 0;
};

struct CutFrame {
    Time time;
    coords3D coords;
    double radar_speed;

    // calculated
    double video_speed = -1;
};

struct CameraInfo {
    double matrix_width_pxl;
    double matrix_height_pxl;
    double matrix_width_mm;
    double matrix_height_mm;
};

class MapFrames {
public:
    MapFrames(const rapidjson::Document &doc);

    MapFrames();

    std::map<std::string, std::vector<CutFrame>>
    fill_structure(const Calc3x4 &c);

public:
    std::vector<Frame> frames;
    double focal_length;
    CameraInfo camera_info;
};

#endif //VSPEED_MAPFRAMES_H
