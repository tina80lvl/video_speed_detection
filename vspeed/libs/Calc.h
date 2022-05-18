#ifndef CALC_H_
#define CALC_H_

#include "Transform.h"
#include <optional>
#include <opencv2/core/core.hpp>
#include <functional>

using namespace std;

struct rt_t {
    cv::Matx33d R;
    cv::Vec3d T;
};

class Calc3x4 {
public:
    using M3x4 = double[3 * 4];

public:
    Calc3x4();

    explicit Calc3x4(const M3x4 &m3x4);

    virtual ~Calc3x4();

    void setMatrix3x4(const double *m3x4_new);

    const double *matrix3x4() const;

    void copyM3x4To(double *dest) const;

    bool toRt(rt_t *rt) const;

    optional<cv::Vec3d> eulerAngles() const;

    bool isValid() const;

    static bool isValid(const double *m3x4);

    void mapRadarToCamera(double x, double y, double z, double *camera_x,
                          double *camera_y) const;

    // отображение позиции номера на картинке в позицию на дороге
    bool
    posBySection(double cam_lx, double cam_ly, double cam_rx, double cam_ry,
                 double w, double *radar_x, double *radar_y, double *radar_lz,
                 double *radar_rz) const;

    void setOnMatrixSet(const function<void()> &handler);

private:
    M3x4 m3x4;
    mutable optional<cv::Vec3d> euler_angles; // градусы
    function<void()> on_matrix_set;
};

class CalcP {
public:
    // cam_teta - угол установки прибора - камера зкреплена жестко соостно с прибором
    // cam_and_radar_teta - поворот радара относитьльно камеры (влево это + (+ - это против часовой стрелки))
    // radar_teta - угол введенный в радар
    CalcP();

    bool isValid() const;

    void setImgSize(double w, double h);

    bool setTrigCoefs(double cam_teta, double radar_teta, double reika_len,
                      int reika_len_in_pixel, double reika_distance,
                      double calib_image_center_by_x_delta);

    bool getTrigCoefs(double *cam_teta, double *radar_teta, double *reika_len,
                      int *reika_len_in_pixel, double *reika_distance,
                      double *calib_image_center_by_x_delta) const;

    void mapRadarToCamera(double ut_x, double ut_y, double ut_len,
                          double physical_lic_width, double *ut_imgx,
                          double *ut_imgy,
                          int *ut_numw) const;

    void calcValues(double ut_x, double ut_y, double ut_len,
                    double physical_lic_width, double *ut_imgx, double *ut_imgy,
                    int *ut_numw, double *ut_imgx_left, double *ut_imgx_right,
                    double *ut_imgy_top, double *ut_imgy_bottom) const;

    void mapCameraToRadar(double camera_x, double camera_y, double numw,
                          double *radar_x, double *radar_y,
                          double physical_lic_width) const;

    static void
    toRadarCoord(double cam_teta, double calib_image_center_by_x_delta,
                 double radar_teta, double reika_len, double reika_len_in_pixel,
                 double reika_distance, double w, double imgx, double numw,
                 double cordon_height, double *radar_x, double *radar_y,
                 double physical_lic_width = 0.5, double licnum_height = 0.5);

    void setCordonHeight(double height);

    void setTackingAngle(double tacking_angle);

    double getMountingAngleInDegrees() const;

public:
    double cam_and_radar_teta; // not used now

    double cam_teta;
    double radar_teta;
    double reika_len;
    double reika_len_in_pixel;
    double reika_distance;

    double image_width;
    double image_height;

    // высота установки прибора от дороги
    double cordon_height;
    // угол наклона прибора к дороге
    double tacking_angle;

    // смещении позиции х пикселя относительно центра видимости прибора
    double calib_image_center_by_x_delta;

    double licnum_height; // высота подвеса номера
};

class CalcS {
public:
    CalcS();

    virtual ~CalcS();

    bool setMatrix(double *m);

    bool getMatrix(double *m) const;

    bool isValid() const;

    bool isEqual(double *m) const;

    bool isNotEqual(double *m) const;

    void mapRadarToCamera(double radar_x, double radar_y,
                          double *camera_x, double *camera_y) const;

    void mapCameraToRadar(double camera_x, double camera_y,
                          double *radar_x, double *radar_y) const;

private:
    Transform t_radar_to_cam;//*trans;
    Transform t_cam_to_radar;//*trans_inv;
};

#endif /* CALC_H_ */
