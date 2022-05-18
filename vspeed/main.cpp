#include <iostream>

#include <filesystem>

#include "utils/utils.h"


int main() {

    std::string path = "/Users/tina/Documents/Masters/diploma/video_speed_detection/rc/parsed/";
    std::ofstream report_file;
    report_file.open("../reports/report.csv");
    report_file
            << "file,frame_dist,targets,frame_per_target,mistake_ms,mistake_kmh\n";

    for (const auto &file: std::filesystem::directory_iterator(path)) {
        auto report = process_file(file.path());
        report_file << report;
    }

    return 0;
}
