#include <iostream>

#include <filesystem>

#include "utils/utils.h"


int main() {
    // Folder with parsed to JSON-format RC-files
    std::string path = "/Users/tina/Documents/Masters/diploma/video_speed_detection/rc/parsed/";

    // File to save statistics
    std::ofstream report_file;
    report_file.open("../reports/report.csv");
    report_file
            << "file,targets,frames_per_target,mistake_ms,mistake_kmh,false_positive\n";

    // File to save all calculated speeds
    std::ofstream speeds_file;
    speeds_file.open("../reports/speeds.csv");
    speeds_file << "frame_dist,licnum,calc_speed,radar_speed\n";

    // File to save all calculated 3D-coordinates
    std::ofstream coords_file;
    coords_file.open("../reports/coords.csv");
    coords_file << "licnum,sec,usec,x,y,zl,radar_speed\n";

    for (const auto &file: std::filesystem::directory_iterator(path)) {
        const auto &report = process_file(file.path(), speeds_file,
                                          coords_file);
        report_file << report;
    }

    report_file.close();
    speeds_file.close();
    coords_file.close();

    return 0;
}
