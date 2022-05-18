#include "Rectangle.h"
#include <opencv2/core/core.hpp>

Rectangle::Rectangle() {
    x = 0;
    y = 0;
    w = 0;
    h = 0;
}

Rectangle::Rectangle(int x, int y, int w, int h) {
    this->x = x;
    this->y = y;
    this->w = w;
    this->h = h;
}

/* изменить ширину, центр остается там же*/
void Rectangle::changeWidth(int new_w) {
    int center_x = x + w / 2;
    x = center_x - new_w / 2;
    w = new_w;
}

/* изменить высоту, центр остается там же*/
void Rectangle::changeHeight(int new_h) {
    int center_y = y + h / 2;
    y = center_y - new_h / 2;
    h = new_h;
}

/* */
void Rectangle::roundExpand(int r_by_x, int r_by_y) {
    x = (x / r_by_x) * r_by_x;
    y = (y / r_by_y) * r_by_y;
    w = ((w + r_by_x - 1) / r_by_x) * r_by_x;
    h = ((h + r_by_y - 1) / r_by_y) * r_by_y;
}

/*
 * расширить прямоугольник так чтобы соотноошение сторон стало needed_coef (w/h),
 * при этом прямоугольник мож только увеличиваться
 * увеличиваем либо h либо w, центр должен остаться там же
 */
void Rectangle::expandToAspectRatio(double needed_coef) {
    double coef = (double) w / (double) h;
    int new_w = w;
    int new_h = h;
    if (coef < needed_coef) { // увеличить ширину
        new_w = (double) h * needed_coef;
    } else {
        new_h = (double) w / needed_coef;
    }
    int center_x = x + w / 2;
    int center_y = y + h / 2;
    x = center_x - new_w / 2;
    y = center_y - new_h / 2;
    w = new_w;
    h = new_h;
}

Rectangle Rectangle::absoluteScaled(double scale) const {
    return {static_cast<int>(x * scale), static_cast<int>(y * scale),
            static_cast<int>(w * scale), static_cast<int>(h * scale)};
}

Rectangle Rectangle::centerScaled(double scale) const {
    return centerScaled(scale, scale);
}

Rectangle
Rectangle::centerScaled(double horizontal_scale, double vertical_scale) const {
    const int new_w = static_cast<int>(w * horizontal_scale);
    const int new_h = static_cast<int>(h * vertical_scale);

    const int delta_w = new_w - w;
    const int delta_h = new_h - h;

    return {x - delta_w / 2, y - delta_h / 2, new_w, new_h};

}

Rectangle Rectangle::intersected(const Rectangle &other) const {
    if (!isIntersectingWith(other)) {
        return {};
    }

    const int x_end = x + w;
    const int y_end = y + h;
    const int other_x_end = other.x + other.w;
    const int other_y_end = other.y + other.h;

    const int new_x = max(x, other.x);
    const int new_y = max(y, other.y);
    const int new_x2 = min(x_end, other_x_end);
    const int new_y2 = min(y_end, other_y_end);

    return {new_x, new_y, new_x2 - new_x, new_y2 - new_y};
}

bool Rectangle::isIntersectingWith(const Rectangle &other) const {
    const int x_end = x + w;
    const int y_end = y + h;
    const int other_x_end = other.x + other.w;
    const int other_y_end = other.y + other.h;

    auto intersects = [](int left1, int right1, int left2, int right2) {
        return left1 <= right2 && right1 >= left2;
    };

    if (!intersects(x, x_end, other.x, other_x_end)
        || !intersects(y, y_end, other.y, other_y_end)) {
        return false;
    }
    return true;
}

string Rectangle::toStr() const {
    char buff[1024];
    sprintf(buff, "%d,%d,%dx%d", x, y, w, h);
    return buff;
}

/*static*/Rectangle Rectangle::fromStr(const string &rect_str) {
    Rectangle r;
    int nf = sscanf(rect_str.c_str(), "%d,%d,%dx%d", &r.x, &r.y, &r.w, &r.h);
    if (nf != 4) {
        r.w = 0;
    }
    return r;
}

/**
 * @param point - точка
 * @return содержится ли точка внутри прямоугольника и на его границах
 */
bool Rectangle::contains(const cv::Point &point) const {
    const int x2 = x + w;
    const int y2 = y + h;

    const bool x_fits = x <= point.x && point.x <= x2;
    const bool y_fits = y <= point.y && point.y <= y2;

    return x_fits && y_fits;
}

cv::Point Rectangle::center() const {
    return cv::Point(x + w / 2, y + h / 2);
}

void Rectangle::moveCenterTo(const cv::Point &point) {
    const auto move_vector = point - center();
    x += move_vector.x;
    y += move_vector.y;
}
