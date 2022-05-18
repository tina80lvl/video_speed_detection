#ifndef OPENCVFWD_H
#define OPENCVFWD_H

#include <cstdint>

namespace cv
{

template<typename _Tp>
class Point_;
typedef Point_<int> Point2i;
typedef Point_<int64_t> Point2l;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;

}

#endif //OPENCVFWD_H
