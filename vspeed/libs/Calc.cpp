#include "Calc.h"
#include "Log.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
//#include "CommonForMatrix3x3.h"
#include <opencv2/calib3d/calib3d.hpp>

//==============================================================================
//====================================== Calc3x4 ===============================
//==============================================================================

Calc3x4::Calc3x4() {
    for (int i = 0; i < 12; i++) {
        m3x4[i] = 0.0;
    }
}

Calc3x4::~Calc3x4() {
}

bool Calc3x4::isValid(const double *m3x4) {
    // last row must be not zero
    if (m3x4[2 * 4 + 0] != 0.0)
        return true;
    if (m3x4[2 * 4 + 1] != 0.0)
        return true;
    if (m3x4[2 * 4 + 2] != 0.0)
        return true;
    if (m3x4[2 * 4 + 3] != 0.0)
        return true;
    return false;
}

bool Calc3x4::isValid() const {
    return Calc3x4::isValid(m3x4);
}

void Calc3x4::setMatrix3x4(const double *m3x4_new) {
    memcpy(m3x4, m3x4_new, sizeof(m3x4));
    euler_angles.reset();
    if (on_matrix_set) {
        on_matrix_set();
    }
}

void Calc3x4::mapRadarToCamera(double x, double y, double z, double *camera_x,
                               double *camera_y) const {
    double xs = m3x4[0 * 4 + 0] * x + m3x4[0 * 4 + 1] * y + m3x4[0 * 4 + 2] * z
                + m3x4[0 * 4 + 3];
    double ys = m3x4[1 * 4 + 0] * x + m3x4[1 * 4 + 1] * y + m3x4[1 * 4 + 2] * z
                + m3x4[1 * 4 + 3];
    double w = m3x4[2 * 4 + 0] * x + m3x4[2 * 4 + 1] * y + m3x4[2 * 4 + 2] * z
               + m3x4[2 * 4 + 3];
    *camera_x = xs / w;
    *camera_y = ys / w;
}

bool Calc3x4::posBySection(double cam_lx, double cam_ly, double cam_rx,
                           double cam_ry, double w, double *radar_x,
                           double *radar_y,
                           double *radar_lz, double *radar_rz) const {
#define P(a, b) (m3x4[(a-1)*4+(b-1)])
    double A[16] = {(P(1, 1) - P(3, 1) * cam_lx), (P(1, 2) - P(3, 2) * cam_lx),
                    (P(1, 3) - P(3, 3) * cam_lx), 0,
                    (P(2, 1) - P(3, 1) * cam_ly),
                    (P(2, 2) - P(3, 2) * cam_ly), (P(2, 3) - P(3, 3) * cam_ly),
                    0,
                    (P(1, 1) - P(3, 1) * cam_rx), (P(1, 2) - P(3, 2) * cam_rx),
                    0,
                    (P(1, 3) - P(3, 3) * cam_rx), (P(2, 1) - P(3, 1) * cam_ry),
                    (P(2,
                       2) - P(3, 2) * cam_ry), 0, (P(2, 3) - P(3, 3) * cam_ry)};
    double B[4] = {
            P(3, 4) * cam_lx - P(1, 4),
            P(3, 4) * cam_ly - P(2, 4), -P(3, 2) * cam_rx * w + P(3, 4) * cam_rx
                                        + P(1, 2) * w - P(1, 4),
            -P(3, 2) * cam_ry * w + P(3, 4) * cam_ry
            + P(2, 2) * w - P(2, 4)};
#undef P
    double A_inv[16];

    bool inv_ok = Transform::inv(A, 4, A_inv);
    if (!inv_ok)
        return false;
    double v[4];
    Transform::mutToV(A_inv, B, 4, v);
    *radar_x = v[0];
    *radar_y = v[1];
    *radar_lz = v[2];
    *radar_rz = v[3];
    return true;
}

void Calc3x4::copyM3x4To(double *dest) const {
    memcpy(dest, m3x4, sizeof(m3x4));
}

const double *Calc3x4::matrix3x4() const {
    return m3x4;
}

optional<cv::Vec3d> Calc3x4::eulerAngles() const {

    if (!euler_angles) {
        rt_t rt;
        const bool current_rt_calculated = toRt(&rt);
//		if (current_rt_calculated) {
//			euler_angles = device_installation_euler_angles_from_R(rt.R);
//		}
    }
    return euler_angles;
}

bool Calc3x4::toRt(rt_t *rt) const {
    if (!isValid(m3x4)) {
        return false;
    }

    cv::Matx34d proj;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            proj(i, j) = m3x4[i * 4 + j];
        }
    }
    cv::Matx33d camera;
    cv::Vec4d trans_vec;
//	decomposeProjectionMatrix(proj, camera, rt->R, trans_vec);
    cv::Vec3d t3(trans_vec[0], trans_vec[1], trans_vec[2]);
    t3 /= trans_vec[3];
    rt->T = -rt->R * t3;
    return true;
}

Calc3x4::Calc3x4(const M3x4 &m3x4) {
    setMatrix3x4(m3x4);
}

void Calc3x4::setOnMatrixSet(const function<void()> &handler) {
    on_matrix_set = handler;
}

//==============================================================================
//====================================== CalcP =================================
//==============================================================================
CalcP::CalcP() {
    this->cam_teta = 0.0;
    this->radar_teta = 0.0;
    this->reika_len = 0.0;
    this->reika_len_in_pixel = 0.0;
    this->reika_distance = 0.0;
    this->calib_image_center_by_x_delta = 0.0;

    this->cordon_height = 0.0; // default
    this->cam_and_radar_teta = 0.0; // not used now
    this->image_width = 0;
    this->image_height = 0;

    // такие значения стояли раньше и никогда не менялись
    this->tacking_angle = 6.0 * M_PI / 180.0; // угол наклона к дороге

    licnum_height = 0.5;
}

bool CalcP::isValid() const {
    if (radar_teta >= M_PI * 2)
        return false;
    if (cam_teta >= M_PI * 2)
        return false;
    return true;
}

void CalcP::setImgSize(double w, double h) {
    if ((this->image_width == w) && (this->image_height == h)) {
        return;
    }
    this->image_width = w;
    this->image_height = h;
}

bool CalcP::setTrigCoefs(double cam_teta, double radar_teta, double reika_len,
                         int reika_len_in_pixel, double reika_distance,
                         double calib_image_center_by_x_delta) {
    LOGC(DEBUG) << "CalcP::setTrigCoefs():\n" << "\tcam_teta " << cam_teta
                << "\n" << "\tradar_teta " << radar_teta << "\n"
                << "\treika_len "
                << reika_len << "\n" << "\treika_len_in_pixel "
                << reika_len_in_pixel << "\n" << "\treika_distance "
                << reika_distance << "\n" << "\tcalib_image_center_by_x_delta "
                << calib_image_center_by_x_delta << std::endl;
    this->cam_teta = cam_teta;
    this->radar_teta = radar_teta;
    this->reika_len = reika_len;
    this->reika_len_in_pixel = (double) reika_len_in_pixel;
    this->reika_distance = reika_distance;
    this->calib_image_center_by_x_delta = calib_image_center_by_x_delta;
    return true;
}

bool CalcP::getTrigCoefs(double *cam_teta, double *radar_teta,
                         double *reika_len, int *reika_len_in_pixel,
                         double *reika_distance,
                         double *calib_image_center_by_x_delta) const {
    *cam_teta = this->cam_teta;
    *radar_teta = this->radar_teta;
    *reika_len = this->reika_len;
    *reika_len_in_pixel = this->reika_len_in_pixel;
    *reika_distance = this->reika_distance;
    *calib_image_center_by_x_delta = this->calib_image_center_by_x_delta;
    return true;
}


void CalcP::mapRadarToCamera(double ut_x, double ut_y, double ut_len,
                             double physical_lic_width, double *ut_imgx,
                             double *ut_imgy,
                             int *ut_numw) const {
    double x_n = ut_x;
    double y_n = ut_y;

    double pixel_in_number =
            reika_len_in_pixel * physical_lic_width / reika_len;

    double a = atan2(y_n, x_n);
    double alpha = a - radar_teta;
    //cam_and_radar_teta not used
    //alpha += cam_and_radar_teta;
    double imgx = tan(alpha) * reika_len_in_pixel * reika_distance / reika_len;
    imgx = imgx + image_width / 2.0;
    *ut_imgx = image_width - imgx;

    double z = cordon_height - licnum_height; // переходим в плоскость номеров
    double a1 = atan2(z, hypot(x_n, y_n));
    double alpha1 = tacking_angle - a1;
    double imgy = tan(alpha1) * reika_len_in_pixel * reika_distance / reika_len;
    imgy = imgy + image_height / 2.0;
    *ut_imgy = image_height - imgy;
//
    double r = hypot(x_n, y_n);
    *ut_numw = (int) (((pixel_in_number * reika_distance) / r) * cos(a));
}

// cam_teta - угол установки прибора - камера зкреплена жестко соостно с прибором
// cam_and_radar_teta - поворот радара относитьльно камеры (влево это + (+ - это против часовой стрелки))
// radar_teta - угол введенный в радар
void CalcP::calcValues(double ut_x, double ut_y, double ut_len,
                       double physical_lic_width, double *ut_imgx,
                       double *ut_imgy,
                       int *ut_numw, double *ut_imgx_left,
                       double *ut_imgx_right,
                       double *ut_imgy_top, double *ut_imgy_bottom) const {
    double x_n = ut_x;
    double y_n = ut_y;

    double pixel_in_number =
            reika_len_in_pixel * physical_lic_width / reika_len;

    double a = atan2(y_n, x_n);
    double r = hypot(x_n, y_n);
    *ut_numw = (int) (((pixel_in_number * reika_distance) / r) * cos(a));

    double alpha = a - radar_teta;
    //cam_and_radar_teta not used
    //alpha += cam_and_radar_teta;
    double imgx = tan(alpha) * reika_len_in_pixel * reika_distance / reika_len;
    imgx = imgx + image_width / 2.0;
    *ut_imgx = image_width - imgx;

    double z = cordon_height - licnum_height; // переходим в плоскость номеров
    double a1 = atan2(z, hypot(x_n, y_n));
    double alpha1 = tacking_angle - a1;
    double imgy = tan(alpha1) * reika_len_in_pixel * reika_distance / reika_len;
    imgy = imgy + image_height / 2.0;
    *ut_imgy = image_height - imgy;

    double a_l = atan2(y_n, x_n + ut_len / 2.0);
    a_l = a_l - radar_teta;
    //cam_and_radar_teta not used
    //a_l += cam_and_radar_teta;
    double imgx_left =
            tan(a_l) * reika_len_in_pixel * reika_distance / reika_len;
    imgx_left = imgx_left + image_width / 2.0;
    *ut_imgx_left = image_width - imgx_left;

    double a_r = atan2(y_n, x_n - ut_len / 2.0);
    a_r = a_r - radar_teta;
    //cam_and_radar_teta not used
    //a_r += cam_and_radar_teta;
    double imgx_right =
            tan(a_r) * reika_len_in_pixel * reika_distance / reika_len;
    imgx_right = imgx_right + image_width / 2.0;
    *ut_imgx_right = image_width - imgx_right;
//

    *ut_imgy_top = *ut_imgy;
    *ut_imgy_bottom = *ut_imgy;
}

void CalcP::mapCameraToRadar(double camera_x, double camera_y, double numw,
                             double *radar_x, double *radar_y,
                             double physical_lic_width) const {
    toRadarCoord(cam_teta, calib_image_center_by_x_delta, radar_teta, reika_len,
                 reika_len_in_pixel, reika_distance, image_width, camera_x,
                 numw,
                 cordon_height, radar_x, radar_y, physical_lic_width,
                 licnum_height);
}

void CalcP::toRadarCoord(double cam_teta, double calib_image_center_by_x_delta,
                         double radar_teta, double reika_len,
                         double reika_len_in_pixel,
                         double reika_distance, double w, double imgx,
                         double numw,
                         double cordon_height, double *radar_x, double *radar_y,
                         double physical_lic_width, double licnum_height) {
//	double visible_angle = (atan( reika_len/reika_distance )/reika_len_in_pixel) *w;
//	double corr = visible_angle/2.0 - imgx*visible_angle/w;

    double alpha = atan((w / 2 + calib_image_center_by_x_delta - imgx)

                        / (reika_len_in_pixel * reika_distance / reika_len));
    //cam_and_radar_teta not used
    //double a = alpha + radar_teta - cam_and_radar_teta;
    double a = alpha + radar_teta;
    double pixel_in_number =
            reika_len_in_pixel * physical_lic_width / reika_len;
    //cam_and_radar_teta not used
//	double r = cos(alpha + cam_teta + cam_and_radar_teta) * pixel_in_number
//			* reika_distance / numw;
    double r = cos(alpha + cam_teta) * pixel_in_number * reika_distance / numw;

    // "опускаемся" на уровень машин(номеров)
    double rh = sqrt(
            r * r
            - (cordon_height - licnum_height)
              * (cordon_height - licnum_height));

    *radar_x = rh * cos(a);
    *radar_y = rh * sin(a);
}

void CalcP::setCordonHeight(double height) {
    LOGC(DEBUG) << "Set new height for CalcP. " << cordon_height << "=>"
                << height << std::endl;
    cordon_height = height;
}

void CalcP::setTackingAngle(double tacking_angle) {
    double old_tacking = this->tacking_angle * 180 / M_PI;
    double new_tacking = tacking_angle * 180 / M_PI;
    LOGC(DEBUG) << "Set new tacking angle for CalcP. " << this->tacking_angle
                << "(" << old_tacking << ")" << "=>" << tacking_angle << "("
                << new_tacking << ")" << std::endl;
    this->tacking_angle = tacking_angle;
}

double CalcP::getMountingAngleInDegrees() const {
    return cam_teta * 180 / M_PI;
}

//==============================================================================
//====================================== CalcS =================================
//==============================================================================
CalcS::CalcS() {
}

CalcS::~CalcS() {
}

bool CalcS::setMatrix(double *m) {
    double m_inv[9];
    if (!Transform::inv(m, 3, m_inv)) {
        LOGIF(ERROR) {
            Transform t_tmp(m);
            printf("can't do inv matrix for :\n");
            t_tmp.dump();
        }
        return false;
    }
    t_radar_to_cam.setMatrix(m);
    t_cam_to_radar.setMatrix(m_inv);
    return true;
}

bool CalcS::getMatrix(double *m) const {
    for (int i = 0; i < 9; i++) {
        m[i] = t_radar_to_cam.matrixPtr()[i];
    }
    return true;
}

bool CalcS::isValid() const {
    if (!t_radar_to_cam.isValid())
        return false;
    if (!t_cam_to_radar.isValid())
        return false;
    return true;
}

bool CalcS::isEqual(double *m) const {
    const double *matrix = t_radar_to_cam.matrixPtr();
    bool ret = true;
    for (int i = 0; i < 9; i++) {
        if (matrix[i] != m[i]) {
            ret = false;
            break;
        }
    }
    return ret;
}

bool CalcS::isNotEqual(double *m) const {
    return !isEqual(m);
}

void CalcS::mapRadarToCamera(double radar_x, double radar_y,
                             double *camera_x, double *camera_y) const {
    t_radar_to_cam.map(radar_x, radar_y, camera_x, camera_y);
}

void CalcS::mapCameraToRadar(double camera_x, double camera_y,
                             double *radar_x, double *radar_y) const {
    t_cam_to_radar.map(camera_x, camera_y, radar_x, radar_y);
}
