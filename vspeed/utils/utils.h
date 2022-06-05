#ifndef VSPEED_UTILS_H
#define VSPEED_UTILS_H

#include <string>
#include <fstream>

std::string process_file(const std::string &path, std::ofstream &speeds_file,
                         std::ofstream &coords_file);

#endif //VSPEED_UTILS_H
