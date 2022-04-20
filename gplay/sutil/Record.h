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
#ifndef __RECORD_H
#define __RECORD_H

#include <optix.h>

#include "cuda/whitted.h"


namespace sutil
{

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};

typedef Record<EmptyData> EmptyRecord;
typedef Record<whitted::HitGroupData> HitGroupRecord;

struct OptixShaderBindingTableWrapped : public OptixShaderBindingTable
{
	OptixShaderBindingTableWrapped()
			: OptixShaderBindingTable()
	{
		raygenRecord = 0;
		exceptionRecord = 0;

		missRecordBase = 0;
		missRecordStrideInBytes = 0;
		missRecordCount = 0;

		hitgroupRecordBase = 0;
		hitgroupRecordStrideInBytes = 0;
		hitgroupRecordCount = 0;
	}

	template<typename RaygenRecordType, typename MissRecordType>
	void allocateRecords(OptixProgramGroup raygen_prog_group, const std::vector<OptixProgramGroup>& miss_groups)
	{
		if(raygenRecord) {
			cudaCheck(cudaFree(reinterpret_cast<void **>(&raygenRecord)));
		}
		cudaCheck(cudaMalloc(reinterpret_cast<void **>(&raygenRecord), sizeof(RaygenRecordType)));
		// TODO действительно надо это делать с начала на хосте?
		RaygenRecordType rg_sbt;
		optixCheck(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
		cudaCheck(cudaMemcpy(
				reinterpret_cast<void*>(raygenRecord),
				&rg_sbt,
				sizeof(RaygenRecordType),
				cudaMemcpyHostToDevice
		));

		if(missRecordBase) {
			cudaCheck(cudaFree(reinterpret_cast<void **>(&missRecordBase)));
		}
		cudaCheck(cudaMalloc(
				reinterpret_cast<void **>(&missRecordBase),
				sizeof(MissRecordType) * miss_groups.size()
		));
		// TODO действительно надо это делать с начала на хосте?
		std::vector<MissRecordType> ms_sbt(miss_groups.size());
		for(size_t i = 0; i < miss_groups.size(); ++i) {
			optixCheck(optixSbtRecordPackHeader(miss_groups[i],  &ms_sbt[i]));
		}

		cudaCheck(cudaMemcpy(
				reinterpret_cast<void*>(missRecordBase),
				ms_sbt.data(),
				sizeof(MissRecordType) * miss_groups.size(),
				cudaMemcpyHostToDevice
		));
		missRecordStrideInBytes = static_cast<unsigned>(sizeof(MissRecordType));
		missRecordCount = miss_groups.size();
	}

	template<typename HitGroupRecordType>
	void allocateHitGroups(const std::vector<HitGroupRecordType>& hit_groups)
	{
		if(hitgroupRecordBase) {
			cudaCheck(cudaFree(reinterpret_cast<void**>(&hitgroupRecordBase)));
		}
		cudaCheck(cudaMalloc(
				reinterpret_cast<void**>(&hitgroupRecordBase),
				sizeof(HitGroupRecordType) * hit_groups.size()
		));
		cudaCheck(cudaMemcpy(
				reinterpret_cast<void*>(hitgroupRecordBase),
				hit_groups.data(),
				sizeof(HitGroupRecordType) * hit_groups.size(),
				cudaMemcpyHostToDevice
		));

		hitgroupRecordStrideInBytes = static_cast<unsigned>(sizeof(HitGroupRecordType));
		hitgroupRecordCount = static_cast<unsigned>(hit_groups.size());
	}
};

} // end namespace sutil

#endif // __RECORD_H
