#ifndef VSPEED_CALCSPEED_H
#define VSPEED_CALCSPEED_H

#define SPD std::map<std::string, std::vector<std::pair<double, double>>>

#include <map>
#include "MapFrames.h"

SPD calculate_speeds(int frame_dist,
                     std::map<std::string, std::vector<CutFrame>> &map);

std::pair<double, double> speed_detection_quality(const SPD &speeds_by_licnum);


#endif //VSPEED_CALCSPEED_H
