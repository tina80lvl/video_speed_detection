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

#include "cuda/whitted.h"
#include "cuda/cuda_utils.h"
#include "cuda/BufferView.h"
#include "optix_utils.h"
#include "cuda/MaterialData.h"


#include "Aabb.h"
#include "Animation.h"
#include "Camera.h"
#include "Matrix.h"
#include "Preprocessor.h"
#include "sutilapi.h"
#include "Record.h"

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>

#include "tinygltf/tiny_gltf.h"

namespace sutil
{

struct SceneNode
{
    int id = -1;
    int parent = -1;
    Matrix4x4 transform;
    bool animated = false;
    bool a = false;

public:
    SceneNode(int node_id, int father)
        : id(node_id),
          parent(father)
    {}
};


class Scene
{
public:
    struct MeshGroup
    {
        std::string name;
        Matrix4x4 transform;

        std::vector<GenericBufferView> indices;
        std::vector<BufferView<float3>> positions;
        std::vector<BufferView<float3>> normals;
        std::vector<BufferView<float2>> texcoords;

        std::vector<int32_t> material_idx;

        OptixTraversableHandle gas_handle = 0;
        CudaDeviceBuffer d_gas_output;

        Aabb object_aabb;
        Aabb world_aabb;
    };

public:
    inline void addCamera(std::shared_ptr<Camera> camera)    { cameras_.push_back(camera); }
    void addMesh(std::shared_ptr<MeshGroup> mesh)    { m_meshes.push_back(std::move(mesh)); }
    void addMaterial(const MaterialData& mtl)    { m_materials.push_back(mtl); }
    void addBuffer(uint64_t buf_size, const void* data);
    bool addImage(const tinygltf::Image& image);
    void addSampler(
                cudaTextureAddressMode address_s,
                cudaTextureAddressMode address_t,
                cudaTextureFilterMode  filter_mode,
                int32_t          image_idx
                );

    CUdeviceptr getBuffer(int32_t buffer_index) const;
    cudaArray_t getImage(int32_t image_index) const;
    cudaTextureObject_t getSampler(int32_t sampler_index) const;

    void finalize(std::string ptx);

    std::shared_ptr<Camera> camera() const;
    std::shared_ptr<Camera> cameraByName(const std::string &name, std::shared_ptr<Camera> default_camera = nullptr) const;

    OptixPipeline pipeline() const { return m_pipeline; }
    const OptixShaderBindingTable* sbt() const { return &sbt_; }
    OptixTraversableHandle traversableHandle() const { return m_ias_handle; }
    sutil::Aabb aabb() const { return m_scene_aabb; }
    OptixDeviceContext context()  { return context_.get(); }
    const std::vector<MaterialData>& materials() const { return m_materials; }

    void createContext();
    void buildMeshAccels( uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT );
    void buildInstanceAccel( int rayTypeCount = whitted::RAY_TYPE_COUNT );

	void rebuildInstanceAccelAsync(cudaStream_t stream, int rayTypeCount = whitted::RAY_TYPE_COUNT);

	void doAnimation(uint64_t time_now_us );

public:
	std::multimap<std::string, std::shared_ptr<MeshGroup>>  mesh_by_node;

private:
	MaterialData readMaterialData(const tinygltf::Material& material) const;
    void processGLTFNode(int node_id, int parent = -1);

	template<typename T>
	BufferView<T> bufferViewFromGLTF(const int32_t accessor_idx);

    void calcNewTransforms(const shared_ptr<SceneNode> &node);

    inline void addSceneNode(int node_id, int parent) { nodes_.emplace_back(make_shared<SceneNode>(node_id, parent)); }
    void addCamera(int node_id);
    void addMesh(int node_id);

    void updateAnimated();
    void findAnimated(int id);

private:
    void createPTXModule(std::string ptx);
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    std::vector<std::shared_ptr<Camera>> cameras_;
    std::vector<std::shared_ptr<MeshGroup>> m_meshes;
    std::vector<MaterialData> m_materials;
    std::vector<CudaDeviceBuffer> buffers_;
    std::vector<std::shared_ptr<CudaTextureObject>> samplers_;
    std::vector<CudaArray> images_;
    sutil::Aabb m_scene_aabb;
    std::unordered_multimap<int, std::shared_ptr<Animation>> animations_;

    std::vector<std::shared_ptr<SceneNode>> nodes_; // все ноды
    std::vector<int> animated_nodes_; // только анимированные
    std::vector<int> root_nodes_; // только корневые


    OptixContext context_;
	OptixShaderBindingTableWrapped sbt_;
    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    OptixPipeline m_pipeline = nullptr;
    OptixModule m_ptx_module = nullptr;

    OptixProgramGroup m_raygen_prog_group = nullptr;
    OptixProgramGroup m_radiance_miss_group = nullptr;
    OptixProgramGroup m_occlusion_miss_group = nullptr;
    OptixProgramGroup m_radiance_hit_group = nullptr;
    OptixProgramGroup m_occlusion_hit_group = nullptr;
    OptixTraversableHandle m_ias_handle = 0;

public:
	OptixBuffer<OptixInstance> d_instances_;
	CudaDeviceBuffer d_temp_buffer_;
	CudaDeviceBuffer d_ias_output_buffer_;
private:
	tinygltf::Model model_;

    uint64_t max_animation_time_ = 1;
    float last_time_ = 0;

public:
	bool loadScene(const std::string& filename);
};

std::shared_ptr<CudaTextureObject> loadFileToTex(const std::string& fn);

} // end namespace sutil

