//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <sutil/Matrix.h>

//------------------------------------------------------------------------------
//
// Quaternion class
//
//------------------------------------------------------------------------------

namespace sutil
{

class Quaternion
{
public:
    Quaternion()
        : q(make_float4(0))
    {}

    Quaternion(const float4& val)
        : q(val)
    {
    }

    Quaternion(float w, float x, float y, float z)
    :
          q(make_float4(x, y, z, w))
    {}

    Quaternion(const float3& from, const float3& to);

    Quaternion(const Quaternion& a)
        : q(a.q)
    {}

    Quaternion ( float angle, const float3& axis );

    // getters and setters
    inline void setW(float _w)       { q.w = _w; }
    inline void setX(float _x)       { q.x = _x; }
    inline void setY(float _y)       { q.y = _y; }
    inline void setZ(float _z)       { q.z = _z; }
    inline float4 getFloat4() const { return q; }
    inline float3 getFloat3() const { return make_float3(q); }
    inline float w() const           { return q.w; }
    inline float x() const           { return q.x; }
    inline float y() const           { return q.y; }
    inline float z() const           { return q.z; }


    Quaternion& operator-=(const Quaternion& r)
    { q -= r.q; return *this; }

    Quaternion& operator+=(const Quaternion& r)
    { q += r.q; return *this; }

    Quaternion& operator*=(const Quaternion& r);

    Quaternion& operator/=(const float a);

    Quaternion conjugate()
    { return Quaternion( q.w, -q.x, -q.y, -q.z ); }

    void rotation( float& angle, float3& axis ) const;
    void rotation( float& angle, float& x, float& y, float& z ) const;
    Matrix4x4 rotationMatrix() const;

    // l2 norm
    float norm() const
    { return sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z); }

    float  normalize();

    float3 eulerAngles() const;
    float3 rotate(const float3 v);

    static Quaternion fromEuler(const float3& v);

private:
    float4 q;
};


inline Quaternion::Quaternion( const float3& from, const float3& to )
{
    const float3 c = cross( from, to );
    q.w = dot(from, to);
    q.x = c.x;
    q.y = c.y;
    q.z = c.z;
}


inline Quaternion::Quaternion( float angle, const float3&  axis )
{
    const float  n        = length( axis );
    const float  inverse  = 1.0f/n;
    const float3 naxis    = axis*inverse;
    const float  s        = sinf(angle/2.0f);

    q.w = naxis.x*s*inverse;
    q.x = naxis.y*s*inverse;
    q.y = naxis.z*s*inverse;
    q.z = cosf(angle/2.0f);
}


inline Quaternion& Quaternion::operator*=(const Quaternion& r)
{

    float w = q.w*r.w() - q.x*r.x() - q.y*r.y() - q.z*r.z();
    float x = q.w*r.x() + q.x*r.w() + q.y*r.z() - q.z*r.y();
    float y = q.w*r.y() + q.y*r.w() + q.z*r.x() - q.x*r.z();
    float z = q.w*r.z() + q.z*r.w() + q.x*r.y() - q.y*r.x();

    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return *this;
}


inline Quaternion& Quaternion::operator/=(const float a)
{
    float inverse = 1.0f/a;
    q.w *= inverse;
    q.x *= inverse;
    q.y *= inverse;
    q.z *= inverse;
    return *this;
}

inline void Quaternion::rotation( float& angle, float3& axis ) const
{
    Quaternion n = *this;
    n.normalize();
    axis = n.getFloat3();
    angle = 2.0f * acosf(n.w());
}

inline void Quaternion::rotation(
        float& angle,
        float& x,
        float& y,
        float& z
        ) const
{
    Quaternion n = *this;
    n.normalize();
    x = n.x();
    y = n.y();
    z = n.z();
    angle = 2.0f * acosf(n.w());
}

inline float Quaternion::normalize()
{
    float n = norm();
    float inverse = 1.0f/n;
    q.w *= inverse;
    q.x *= inverse;
    q.y *= inverse;
    q.z *= inverse;
    return n;
}


inline Quaternion operator*(const float a, const Quaternion &r)
{ return Quaternion(r.getFloat4() * a); }


inline Quaternion operator*(const Quaternion &r, const float a)
{ return Quaternion(r.getFloat4() * a); }


inline Quaternion operator/(const Quaternion &r, const float a)
{
    float inverse = 1.0f / a;
    return Quaternion(r.getFloat4() * inverse);
}


inline Quaternion operator/(const float a, const Quaternion &r)
{
    float inverse = 1.0f/a;
    return Quaternion(r.getFloat4() * inverse);
}


inline Quaternion operator-(const Quaternion& l, const Quaternion& r)
{ return Quaternion(l.getFloat4() - r.getFloat4()); }


inline bool operator==(const Quaternion& l, const Quaternion& r)
{ return l.getFloat4() == r.getFloat4(); }


inline bool operator!=(const Quaternion& l, const Quaternion& r)
{ return !(l == r); }


inline Quaternion operator+(const Quaternion& l, const Quaternion& r)
{ return Quaternion(l.getFloat4() + r.getFloat4()); }


inline Quaternion operator*(const Quaternion& l, const Quaternion& r)
{
    float w = l.w()*r.w() - l.x()*r.x() - l.y()*r.y() - l.z()*r.z();
    float x = l.w()*r.x() + l.x()*r.w() + l.y()*r.z() - l.z()*r.y();
    float y = l.w()*r.y() + l.y()*r.w() + l.z()*r.x() - l.x()*r.z();
    float z = l.w()*r.z() + l.z()*r.w() + l.x()*r.y() - l.y()*r.x();
    return Quaternion( w, x, y, z );
}

inline float dot( const Quaternion& l, const Quaternion& r )
{
    return l.w()*r.w() + l.x()*r.x() + l.y()*r.y() + l.z()*r.z();
}


inline Matrix4x4 Quaternion::rotationMatrix() const
{
    Matrix4x4 m;

    const float qw = q.w;
    const float qx = q.x;
    const float qy = q.y;
    const float qz = q.z;

    m[0*4+0] = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;
    m[0*4+1] = 2.0f*qx*qy - 2.0f*qz*qw;
    m[0*4+2] = 2.0f*qx*qz + 2.0f*qy*qw;
    m[0*4+3] = 0.0f;

    m[1*4+0] = 2.0f*qx*qy + 2.0f*qz*qw;
    m[1*4+1] = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;
    m[1*4+2] = 2.0f*qy*qz - 2.0f*qx*qw;
    m[1*4+3] = 0.0f;

    m[2*4+0] = 2.0f*qx*qz - 2.0f*qy*qw;
    m[2*4+1] = 2.0f*qy*qz + 2.0f*qx*qw;
    m[2*4+2] = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;
    m[2*4+3] = 0.0f;

    m[3*4+0] = 0.0f;
    m[3*4+1] = 0.0f;
    m[3*4+2] = 0.0f;
    m[3*4+3] = 1.0f;

    return m;
}

} // end namespace sutil
