#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include <vector>

class Transform {
public:
    Transform(double m11, double m12, double m13, double m21, double m22,
              double m23, double m31, double m32, double m33);

    Transform(const double *m);

    Transform();

    void map(double radar_x, double radar_y, double *camera_x,
             double *camera_y) const;

    bool inv(double *m_inv) const;

    void dump() const;

    const double *matrixPtr() const;

    void setMatrix(const double *m);

    bool isValid() const;

    static bool isValid(const double *m3x3);

    static bool inv(const double *m, int r, double *m_inv);

    static void trans(const double *m, int r, double *m_trans);

    static double detrm(const double *a, int k);

    static void cofact(const double *num, int f, double *fac);

    static void mul(const double *A, const double *B, int r, double *C);

    static bool quadToQuad(const double (&one)[4][2], const double (&two)[4][2],
                           double *res);

    static bool basis(const double (&one)[4][2], int r, double *res);

    static void mutToV(const double *A, const double *v, int r, double *v_out);

private:
    double m[3 * 3];
};

#endif /* TRANSFORM_H_ */
