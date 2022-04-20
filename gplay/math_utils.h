#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>

inline float rad2deg(float a)
{
	return a * 180 / M_PI;
}

inline double rad2deg(double a)
{
	return a * 180 / M_PI;
}

inline float deg2rad(float a)
{
	return a * M_PI / 180;
}

inline double deg2rad(double a)
{
	return a * M_PI / 180;
}

#endif // MATH_UTILS_H
