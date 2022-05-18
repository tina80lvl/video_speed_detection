#include "Transform.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

Transform::Transform() {
    m[0 * 3 + 0] = 0;
    m[0 * 3 + 1] = 0;
    m[0 * 3 + 2] = 0;
    m[1 * 3 + 0] = 0;
    m[1 * 3 + 1] = 0;
    m[1 * 3 + 2] = 0;
    m[2 * 3 + 0] = 0;
    m[2 * 3 + 1] = 0;
    m[2 * 3 + 2] = 0;
}

Transform::Transform(double m11, double m12, double m13, double m21, double m22,
                     double m23, double m31, double m32, double m33) {
    m[0 * 3 + 0] = m11;
    m[0 * 3 + 1] = m12;
    m[0 * 3 + 2] = m13;
    m[1 * 3 + 0] = m21;
    m[1 * 3 + 1] = m22;
    m[1 * 3 + 2] = m23;
    m[2 * 3 + 0] = m31;
    m[2 * 3 + 1] = m32;
    m[2 * 3 + 2] = m33;
}

Transform::Transform(const double *coefs) {
    m[0] = coefs[0];
    m[1] = coefs[1];
    m[2] = coefs[2];
    m[3] = coefs[3];
    m[4] = coefs[4];
    m[5] = coefs[5];
    m[6] = coefs[6];
    m[7] = coefs[7];
    m[8] = coefs[8];
}

void Transform::map(double radar_x, double radar_y, double *camera_x,
                    double *camera_y) const {
    double x = radar_x * m[0 * 3 + 0] + radar_y * m[1 * 3 + 0] + m[2 * 3 + 0];
    double y = radar_x * m[0 * 3 + 1] + radar_y * m[1 * 3 + 1] + m[2 * 3 + 1];
    double w = radar_x * m[0 * 3 + 2] + radar_y * m[1 * 3 + 2] + m[2 * 3 + 2];
    *camera_x = x / w;
    *camera_y = y / w;
}

bool Transform::inv(double *m_inv) const {
    return inv(m, 3, m_inv);
}

void Transform::dump() const {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%20.5lf ", m[i * 3 + j]);
        }
        printf("\n");
    }
}

const double *Transform::matrixPtr() const {
    return m;
}

void Transform::setMatrix(const double *new_m) {
    for (unsigned i = 00; i < 9; i++) {
        m[i] = new_m[i];
    }
}

bool Transform::inv(const double *m, int r, double *m_inv) {
    double d = detrm(m, r);
    if (d == 0.0) {
        return false;
    }
    double fac[r * r];
    cofact(m, r, fac);

    double b[r * r];
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            b[i * r + j] = fac[j * r + i];
        }
    }
//	inv[ i ][ j ] = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            m_inv[i * r + j] = b[i * r + j] / d;
        }
    }
    return true;
}

void Transform::trans(const double *m, int r, double *m_trans) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            m_trans[j + i * r] = m[i + j * r];
        }
    }
}

double Transform::detrm(const double *a, int k) {
    if (k == 1)
        return (a[0 * k + 0]);

    double s = 1.0;
    double det = 0.0;
    for (int c = 0; c < k; c++) {
        double b[(k - 1) * (k - 1)];
        for (int i = 1; i < k; i++) {
            int n = 0;
            for (int j = 0; j < k; j++) {
                if (j != c) {
                    b[(i - 1) * (k - 1) + n] = a[i * k + j];
                    n++;
                }
            }
        }
        det = det + s * (a[0 * k + c] * detrm(b, k - 1));
        s = -1.0 * s;
    }
    return det;
}

void Transform::cofact(const double *num, int f, double *fac) {
    for (int q = 0; q < f; q++) {
        for (int p = 0; p < f; p++) {
            double b[(f - 1) * (f - 1)];
            int m = 0;
            for (int i = 0; i < f; i++) {
                if (i == q) {
                    continue;
                }
                int n = 0;
                for (int j = 0; j < f; j++) {
                    if (j != p) {
                        b[m * (f - 1) + n] = num[i * f + j];
                        n++;
                    }
                }
                m++;
            }
            double sign = ((q + p) % 2 == 1 ? -1.0 : 1.0);
            fac[q * f + p] = sign * detrm(b, f - 1);
        }
    }
}

// C=A*B 
void Transform::mul(const double *A, const double *B, int r, double *C) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            C[i * r + j] = 0.0;
            for (int k = 0; k < r; k++) {
                C[i * r + j] += A[i * r + k] * B[k * r + j];
            }
        }
    }
}

// v_out=A*v 
void Transform::mutToV(const double *A, const double *v, int r, double *v_out) {
    for (int i = 0; i < r; i++) {
        v_out[i] = 0.0;
        for (int k = 0; k < r; k++) {
            v_out[i] += A[i * r + k] * v[k];
        }
    }
}

bool Transform::basis(const double (&one)[4][2], int r, double *res) {
    double m[r * r];
    for (int i = 0; i < (r - 1); i++) {
        for (int j = 0; j < r; j++) {
            m[i * r + j] = one[j][i];
        }
    }
    for (int j = 0; j < r; j++) {
        m[(r - 1) * r + j] = 1.0;
    }
    double m_inv[r * r];
    if (!Transform::inv(m, r, m_inv))
        return false;

    double v[r];
    for (int j = 0; j < (r - 1); j++) {
        v[j] = one[r][j];
    }
    v[r - 1] = 1.0;
    double v_out[r];
    // mul m_inv to col (x4 y4 1) => solve
    mutToV(m_inv, v, r, v_out);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            res[i * r + j] = m[i * r + j] * v_out[j];
        }
    }
    return true;
}

// http://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript
bool Transform::quadToQuad(const double (&one)[4][2], const double (&two)[4][2],
                           double *res) {
    double A[9];
    if (!basis(one, 3, A)) {
        return false;
    }
    double B[9];
    if (!basis(two, 3, B)) {
        return false;
    }

    double A_inv[9];
    if (!Transform::inv(A, 3, A_inv)) {
        return false;
    }
    double C[9];
    mul(B, A_inv, 3, C);

    // в Transform мы используем транспонированную матрицу
    Transform::trans(C, 3, res);
    return true;
}

bool Transform::isValid(const double *m3x3) {
    // last column must be not zero
    if (m3x3[0 * 3 + 2] != 0.0)
        return true;
    if (m3x3[1 * 3 + 2] != 0.0)
        return true;
    if (m3x3[2 * 3 + 2] != 0.0)
        return true;
    return false;
}

bool Transform::isValid() const {
    return isValid(m);
}
