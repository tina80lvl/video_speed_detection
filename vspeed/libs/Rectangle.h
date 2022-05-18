#ifndef PHOTORADARSOURCES_RECTANGLE_H
#define PHOTORADARSOURCES_RECTANGLE_H

#include <string>
#include "OpenCVFwd.h"

using namespace std;

struct [[nodiscard]] Rectangle
{
	Rectangle();
	Rectangle(int x, int y, int w, int h);

	bool operator==(const Rectangle &other) const
	{
		return x == other.x && y == other.y && w == other.w && h == other.h;
	}

	bool operator!=(const Rectangle &other) const
	{
		return !(other == *this);
	}

	bool contains(const cv::Point &point) const;

	/* изменить ширину, центр остается там же*/
	void changeWidth(int new_w);
	/* изменить высоту, центр остается там же*/
	void changeHeight(int new_h);
	/* */
	void roundExpand(int r_by_x, int r_by_y);
	/*
	 * расширить прямоугольник так чтобы соотноошение сторон стало needed_coef (w/h),
	 * при этом прямоугольник мож только увеличиваться
	 * увеличиваем либо h либо w, центр должен остаться там же
	 */
	void expandToAspectRatio(double needed_coef);
	Rectangle absoluteScaled(double scale) const;
	Rectangle centerScaled(double scale) const;
	Rectangle
	centerScaled(double horizontal_scale, double vertical_scale) const;
	Rectangle intersected(const Rectangle &other) const;
	bool isIntersectingWith(const Rectangle &other) const;

	cv::Point center() const;
	void moveCenterTo(const cv::Point &point);

	string toStr() const;
	static Rectangle fromStr(const string &rect_str);

	int x;
	int y;
	int w;
	int h;
};

#endif //PHOTORADARSOURCES_RECTANGLE_H
