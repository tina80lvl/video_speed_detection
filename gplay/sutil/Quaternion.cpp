#include "Quaternion.h"
#include "math_utils.h"


float3 sutil::Quaternion::eulerAngles() const
{
    float3 ang;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    ang.x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        ang.y = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        ang.y = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    ang.z = std::atan2(siny_cosp, cosy_cosp);

    return ang;
}

float3 sutil::Quaternion::rotate(const float3 v)
{
    // Extract the vector part of the quaternion
    float3 u = getFloat3();

    // Extract the scalar part of the quaternion
    float s = q.w;

    // Do the math
    float3 ret = 2.0f * dot(u, v) * u
            + (s*s - dot(u, u)) * v
            + 2.0f * s * cross(u, v);
    return ret;
}

sutil::Quaternion sutil::Quaternion::fromEuler(const float3 &v)
{
    // Abbreviations for the various angular functions
    float cy = cosf(v.z * 0.5);
    float sy = sinf(v.z * 0.5);
    float cp = cosf(v.y * 0.5);
    float sp = sinf(v.y * 0.5);
    float cr = cosf(v.x * 0.5);
    float sr = sinf(v.x * 0.5);

    float4 q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return Quaternion(q);
}
