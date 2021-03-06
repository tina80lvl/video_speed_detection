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

#include <vector_types.h>
#include <cstdint>

#include <optix.h>

#include "BufferView.h"
#include "GeometryData.h"
#include "Light.h"
#include "MaterialData.h"

namespace whitted
{

const uint32_t NUM_PAYLOAD_VALUES = 6u;


struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};


enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
};

struct ParallelogramLight
{
	float3 corner;
	float3 v1, v2;
	float3 normal;
	float3 emission;
};

struct LaunchParams
{
    uint32_t                 subframe_index;
    float4*                  accum_buffer;
	unsigned char*           frame_buffer;
	uint16_t*				 dist_buffer;
    int32_t                  max_depth;

    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;

    BufferView<Light::Point> lights;
	ParallelogramLight     plight;
    float3                   miss_color;
    OptixTraversableHandle   handle;

	cudaTextureObject_t env_tex;

	uint16_t *hall_to_img_buffer;
	float img_3x4[12];
};


struct PayloadRadiance
{
    float3 color;
//    float  importance;
//    int    depth;
	float3 hit_point;
};


struct PayloadOcclusion
{
};


} // end namespace whitted
