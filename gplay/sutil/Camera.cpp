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


#include "sutil/Camera.h"
#include "sutil/Quaternion.h"

#include "math_utils.h"
#include "Log.h"

namespace sutil {

void Camera::setTransform(const Matrix4x4 &transform)
{
    node_xform = transform;
    m_eye = make_float3(transform * float4({0, 0, 0, 1}));
    m_up = make_float3(transform * float4({0, 1, 0, 0}));
    m_lookat = make_float3(0);
    float3 direction = make_float3(transform * float4({0, 0, -1, 0}));
    setDirection(direction);
}

Camera::Camera(const std::string& name, const Matrix4x4 &transform, float fovY, float aspect_ratio)
    : name(name),
      m_fovY(fovY),
      m_aspectRatio(aspect_ratio)
{
    setTransform(transform);
}

void Camera::UVWFrame(float3& U, float3& V, float3& W) const
{
    W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
    float wlen = length(W);
    U = normalize(cross(W, m_up));
    V = normalize(cross(U, W));

    float vlen = wlen * tanf(0.5f * deg2rad(m_fovY));
    V *= vlen;
    float ulen = vlen * m_aspectRatio;
    U *= ulen;
}

float3 Camera::getEulerAngles() const
{
    float3 res;

    auto row0 = node_xform.getRow(0);
    auto row1 = node_xform.getRow(1);
    auto row2 = node_xform.getRow(2);

    if(row0.x == 1 || row0.x == -1) {
        res.x = 0;
        res.y = std::atan2(row0.z, row2.w);
        res.z = 0;
    } else {
        res.x = std::atan2(-row1.z, row1.y);
        res.y = std::atan2(-row2.x, row0.x);
        res.z = std::asin(row1.x);
    }

    return res;
}

void Camera::rotateEulerAngles(float3 angles)
{
    float3 rotate_angles = angles - getEulerAngles();
    auto rotate = sutil::Quaternion::fromEuler(rotate_angles).rotationMatrix();
    setTransform(node_xform * rotate);
}

} // namespace sutil
