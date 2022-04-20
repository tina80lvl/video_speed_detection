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
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_MIRRORED_REPEAT 0x8370
#define GL_NEAREST 0x2600


#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "cuda/whitted.h"

#include "gltf_helpers.h"

#include "math_utils.h"

#include "Scene.h"
#include "Matrix.h"
#include "Quaternion.h"

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <Log.h>

using namespace sutil;

float3 make_float3_from_double( double x, double y, double z )
{
    return make_float3( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ) );
}

template<typename T>
float3 make_float3(const std::vector<T>& vec)
{
    if(vec.size() != 3) {
        LOGC(ERROR) << "Vector not equal to float3. Got: " << vec.size() << std::endl;
        exit(1);
    }
    return { static_cast<float>(vec[0]), static_cast<float>(vec[1]), static_cast<float>(vec[2]) };
}

template<typename T>
float4 make_float4(const std::vector<T>& vec)
{
    if(vec.size() != 4) {
        LOGC(ERROR) << "Vector not equal to float4. Got: " << vec.size() << std::endl;
        exit(1);
    }
    return { static_cast<float>(vec[0]), static_cast<float>(vec[1]), static_cast<float>(vec[2]), static_cast<float>(vec[3]) };
}

float4 make_float4_from_double( double x, double y, double z, double w )
{
    return make_float4( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ), static_cast<float>( w ) );
}

void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    if(level == 0) {
        return;
    }

    std::stringstream str;
    str << "[" << std::setw( 12 ) << tag << "]: " << message << "\n";
    switch (level) {
        case 1:
        case 2:
            LOGC(DEBUG) << str.str();
            break;
        case 3:
            LOGC(WARNING) << str.str();
            break;
        default:
            LOGC(DEBUG) << str.str();
    }
}

template<typename T>
BufferView<T> Scene::bufferViewFromGLTF(const int32_t accessor_idx)
{
    if( accessor_idx == -1 ) {
        return BufferView<T>();
    }

    const auto& gltf_accessor    = model_.accessors[ accessor_idx ];
    const auto& gltf_buffer_view = model_.bufferViews[ gltf_accessor.bufferView ];

    const int32_t elmt_byte_size = gltfComponentType2TypeSize(gltf_accessor.componentType);
    if(!elmt_byte_size) {
        //throw Exception("gltf accessor component type not supported");
        LOGC(ERROR) << "gltf accessor component type " << gltf_accessor.componentType << " not supported " << std::endl;
        exit(1);
    }

    const CUdeviceptr buffer_base = getBuffer( gltf_buffer_view.buffer );
    BufferView<T> buffer_view;
    buffer_view.data           = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
    buffer_view.byte_stride    = static_cast<uint16_t>( gltf_buffer_view.byteStride );
    buffer_view.count          = static_cast<uint32_t>( gltf_accessor.count );
    buffer_view.elmt_byte_size = static_cast<uint16_t>( elmt_byte_size );

    uint64_t v = (uint64_t) buffer_view.data;
    if (v & 7ULL) { // hack for 716 (cudaErrorMisalignedAddress)
        static int cnt;
        cnt++;
        LOGC(WARNING) << "unnaligned data ("
                      << std::hex << buffer_view.data << std::dec
                      << "), remap " << cnt
                      << std::endl;
        cudaCheck(cudaMalloc((void **) &buffer_view.data, buffer_view.count*buffer_view.elmt_byte_size));
        cudaCheck(cudaMemcpy((void *) buffer_view.data,
                (void *) (buffer_base + gltf_buffer_view.byteOffset
                        + gltf_accessor.byteOffset), buffer_view.count*buffer_view.elmt_byte_size,
                cudaMemcpyDeviceToDevice));
        return buffer_view;
    }

    return buffer_view;
}


void Scene::processGLTFNode(int node_id, int parent)
{
    if(static_cast<size_t>(node_id) <  nodes_.size()) {
        nodes_[node_id]->parent = parent;
        return;
    }
    const auto& gltf_node = model_.nodes[node_id];

    LOGC(DEBUG) << "Processing glTF node: '" << gltf_node.name << " clds: " << gltf_node.children.size() << "'\n";

    addSceneNode(node_id, parent);

    if(gltf_node.camera != -1) {
        addCamera(node_id);
    } else if( gltf_node.mesh != -1 ) {
        addMesh(node_id);
    }

    for(int32_t child : gltf_node.children) {
        processGLTFNode(child, node_id);
    }
}

cudaTextureAddressMode gltfAddressMode2Cuda(int mode)
{
    switch(mode) {
        case GL_CLAMP_TO_EDGE:
            return cudaAddressModeClamp;
        case GL_MIRRORED_REPEAT:
            return cudaAddressModeMirror;
        default:
            return cudaAddressModeWrap;
    }
}

cudaTextureFilterMode gltfFilterMode2Cuda(int mode)
{
    if(mode == GL_NEAREST) {
        return cudaFilterModePoint;
    }

    return cudaFilterModeLinear;
}

MaterialData Scene::readMaterialData(const tinygltf::Material& material) const
{
    MaterialData mtl;
    {
        const auto base_color_it = material.values.find( "baseColorFactor" );
        if(base_color_it != material.values.end()) {
            const tinygltf::ColorValue c = base_color_it->second.ColorFactor();
            mtl.base_color = make_float4_from_double( c[0], c[1], c[2], c[3] );
            LOGC(DEBUG)
                << "\tBase color: ("
                << mtl.base_color.x << ", "
                << mtl.base_color.y << ", "
                << mtl.base_color.z << ")\n";
        } else {
            LOGC(DEBUG) << "\tUsing default base color factor\n";
        }
    }

    {
        const auto base_color_it = material.values.find( "baseColorTexture" );
        if( base_color_it != material.values.end() ) {
            LOGC(DEBUG) << "\tFound base color tex: " << base_color_it->second.TextureIndex() << "\n";
            mtl.base_color_tex = getSampler(base_color_it->second.TextureIndex());
        }
        else
        {
            LOGC(DEBUG) << "\tNo base color tex\n";
        }
    }

    {
        const auto roughness_it = material.values.find( "roughnessFactor" );
        if(roughness_it != material.values.end()) {
            mtl.roughness = static_cast<float>( roughness_it->second.Factor() );
            LOGC(DEBUG) << "\tRougness:  " << mtl.roughness <<  "\n";
        } else {
            LOGC(DEBUG) << "\tUsing default roughness factor\n";
        }
    }

    {
        const auto metallic_it = material.values.find( "metallicFactor" );
        if(metallic_it != material.values.end()) {
            mtl.metallic = static_cast<float>( metallic_it->second.Factor() );
            LOGC(DEBUG) << "\tMetallic:  " << mtl.metallic <<  "\n";
        } else {
            LOGC(DEBUG) << "\tUsing default metallic factor\n";
        }
    }

    {
        const auto metallic_roughness_it = material.values.find( "metallicRoughnessTexture" );
        if( metallic_roughness_it != material.values.end() ) {
            LOGC(DEBUG) << "\tFound metallic roughness tex: " << metallic_roughness_it->second.TextureIndex() << "\n";
            mtl.metallic_roughness_tex = getSampler( metallic_roughness_it->second.TextureIndex() );
        } else {
            LOGC(DEBUG) << "\tNo metallic roughness tex\n";
        }
    }

    {
        const auto normal_it = material.additionalValues.find( "normalTexture" );
        if( normal_it != material.additionalValues.end() ) {
            LOGC(DEBUG) << "\tFound normal color tex: " << normal_it->second.TextureIndex() << "\n";
            mtl.normal_tex = getSampler( normal_it->second.TextureIndex() );
        } else {
            LOGC(DEBUG) << "\tNo normal tex\n";
        }
    }
    return mtl;
}

bool Scene::loadScene(const std::string &filename)
{
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret;
    if (filename.substr(filename.find_last_of('.') + 1) == "glb") {
        ret = loader.LoadBinaryFromFile(&model_, &err, &warn, filename);
    } else {
        ret = loader.LoadASCIIFromFile(&model_, &err, &warn, filename);
    }

    if(!warn.empty()) {
        LOGC(WARNING) << "glTF WARNING: " << warn << std::endl;
    }
    if(!ret) {
        LOGC(ERROR) << "Failed to load GLTF scene '" << filename << "': " << err << std::endl;
        return false;
    }

    //
    // Process buffer data first -- buffer views will reference this list
    //
    for(const auto& gltf_buffer : model_.buffers) {
        const uint64_t buf_size = gltf_buffer.data.size();
        LOGC(DEBUG) << "Processing glTF buffer '" << gltf_buffer.name << "'\n"
                  << "\tbyte size: " << buf_size << "\n"
                  << "\turi      : " << gltf_buffer.uri << std::endl;

        addBuffer(buf_size, gltf_buffer.data.data());
    }

    //
    // Images -- just load all up front for simplicity
    //
    for(const auto& gltf_image : model_.images) {
        LOGC(DEBUG) << "Processing image '" << gltf_image.name << "'\n"
                  << "\t(" << gltf_image.width << "x" << gltf_image.height << ")x" << gltf_image.component << "\n"
                  << "\tbits: " << gltf_image.bits << std::endl;

        if (!addImage(gltf_image)) {
            return false;
        }
    }

    //
    // Textures -- refer to previously loaded images
    //
    for(const auto& gltf_texture : model_.textures) {
        if(gltf_texture.sampler == -1) {
            addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, gltf_texture.source);
            continue;
        }

        const auto& gltf_sampler = model_.samplers[gltf_texture.sampler];

        const cudaTextureAddressMode address_s = gltfAddressMode2Cuda(gltf_sampler.wrapS);
        const cudaTextureAddressMode address_t = gltfAddressMode2Cuda(gltf_sampler.wrapT);
        const cudaTextureFilterMode  filter = gltfFilterMode2Cuda(gltf_sampler.minFilter);
        addSampler(address_s, address_t, filter, gltf_texture.source);
    }

    //
    // Materials
    //
    for(auto& gltf_material : model_.materials) {
        LOGC(DEBUG) << "Processing glTF material: '" << gltf_material.name << "'\n";
        addMaterial(readMaterialData(gltf_material));
    }

    //
    // Animations
    //
    for(auto& gltf_animation : model_.animations) {
        LOGC(DEBUG) << "Processing glTF animation: '" << gltf_animation.name << "'\n";
        auto animation = std::make_shared<Animation>(gltf_animation, model_);
        double max_time = static_cast<double>(animation->timeMinMax().second) * 1e6;
        max_animation_time_ = std::max(max_animation_time_, static_cast<uint64_t>(max_time));
        for(const auto& node : animation->getNodes()) {
            animations_.insert({ node, animation });
        }
    }

    //
    // Process nodes
    //
    for(int i = 0; i < static_cast<int>(model_.nodes.size()); ++i) {
        processGLTFNode(i);
    }
    // Выделяем корневые ноды объектов и рассчитываем для них трансформы.
    for(auto& node : nodes_) {
        if(node->parent < 0) {
            root_nodes_.push_back(node->id);
            calcNewTransforms(node);
        }
    }
    updateAnimated(); // нужны корневые ноды

    return true;
}

void Scene::addBuffer(uint64_t buf_size, const void* data)
{
    buffers_.emplace_back(buf_size);
    buffers_.back().copyFromHost(data, buf_size);
}


bool Scene::addImage(const tinygltf::Image& image)
{
    assert(image.component == 4);

    // Allocate CUDA array in device memory
    int32_t pitch;
    CudaArray cuda_image;
    if(image.bits == 8) {
        pitch = image.width * image.component * sizeof(uint8_t);
        cuda_image.allocate<uchar4>(image.width, image.height);
    } else if(image.bits == 16) {
        pitch = image.width * image.component * sizeof(uint16_t);
        cuda_image.allocate<ushort4>(image.width, image.height);
    } else {
        LOGC(ERROR) <<  "Unsupported bits/component in glTF image" << std::endl;
        return false;
    }

    cuda_image.copyFromHost(image.image.data(), pitch);
    images_.emplace_back(std::move(cuda_image));
    return true;
}

void Scene::addSampler(
         cudaTextureAddressMode address_s,
         cudaTextureAddressMode address_t,
         cudaTextureFilterMode filter,
         int32_t image_idx
         )
{
    cudaResourceDesc res_desc = {};
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = getImage(image_idx);

    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = address_s;
    tex_desc.addressMode[1]      = address_t;
    tex_desc.filterMode          = filter;
    tex_desc.readMode            = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 1.0f;
    tex_desc.sRGB                = 0; // TODO: glTF assumes sRGB for base_color -- handle in shader

    samplers_.emplace_back(std::make_shared<CudaTextureObject>(&res_desc, &tex_desc));
}


inline CUdeviceptr Scene::getBuffer( int32_t buffer_index ) const
{
    return reinterpret_cast<CUdeviceptr>(buffers_[buffer_index].get());
}


inline cudaArray_t Scene::getImage(int32_t image_index) const
{
    return images_[image_index].get();
}


inline cudaTextureObject_t Scene::getSampler(int32_t sampler_index) const
{
    return samplers_[sampler_index]->get();
}

void Scene::finalize(std::string ptx)
{
    context_.create();

    {
        // info
        LOGC(DEBUG) << "RT core version: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_RTCORE_VERSION) << std::endl;
        LOGC(DEBUG) << "RT max trace depth: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH) << std::endl;
        LOGC(DEBUG) << "RT max traversable graph depth: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH) << std::endl;
        LOGC(DEBUG) << "RT max primitives per GAS: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS) << std::endl;
        LOGC(DEBUG) << "RT max instance per IAS: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS) << std::endl;
        LOGC(DEBUG) << "RT limit max instance ID: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID) << std::endl;
        LOGC(DEBUG) << "RT num bits instance visibility mask: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK) << std::endl;
        LOGC(DEBUG) << "RT max SBT records per GAS: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS) << std::endl;
        LOGC(DEBUG) << "RT max SBT offset: " << context_.getProperty(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET) << std::endl;
    }

    buildMeshAccels();
    buildInstanceAccel();
    createPTXModule(ptx);
    createProgramGroups();
    createPipeline();
    createSBT();

    m_scene_aabb.invalidate();
    for( const auto mesh: m_meshes )
        m_scene_aabb.include( mesh->world_aabb );
}

std::shared_ptr<Camera> sutil::Scene::camera() const
{
    if(!cameras_.empty()) {
        LOGC(DEBUG) << "Returning first camera" << std::endl;
        return cameras_.front();
    }

    LOGC(DEBUG) << "Returning default camera" << std::endl;
    auto cam = make_shared<Camera>();
    cam->setFovY(45.0f);
    cam->setLookat(m_scene_aabb.center());
    cam->setEye(m_scene_aabb.center() + make_float3(0.0f, 0.0f, 1.5f * m_scene_aabb.maxExtent()));
    return cam;
}

std::shared_ptr<sutil::Camera> sutil::Scene::cameraByName(const std::string &name, std::shared_ptr<sutil::Camera> default_camera ) const
{
    for (auto& c: cameras_) {
        if (c->name == name)
            return c;
    }
    return default_camera;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

void Scene::buildMeshAccels( uint32_t triangle_input_flags )
{
    // Problem:
    // The memory requirements of a compacted GAS are unknown prior to building the GAS.
    // Hence, compaction of a GAS requires to build the GAS first and allocating memory for the compacted GAS afterwards.
    // This causes a device-host synchronization point, potentially harming performance.
    // This is most likely the case for small GASes where the actual building and compaction of the GAS is very fast.
    // A naive algorithm processes one GAS at a time with the following steps:
    // 1. compute memory sizes for the build process (temporary buffer size and build buffer size)
    // 2. allocate temporary and build buffer
    // 3. build the GAS (with temporary and build buffer) and compute the compacted size
    // If compacted size is smaller than build buffer size (i.e., compaction is worth it):
    // 4. allocate compacted buffer (final output buffer)
    // 5. compact GAS from build buffer into compact buffer
    //
    // Idea of the algorithm:
    // Batch process the building and compaction of multiple GASes to avoid host-device synchronization.
    // Ideally, the number of synchronization points would be linear with the number of batches rather than the number of GASes.
    // The main constraints for selecting batches of GASes are:
    // a) the peak memory consumption when batch processing GASes, and
    // b) the amount of memory for the output buffer(s), containing the compacted GASes. This is also part of a), but is also important after the build process.
    // For the latter we try to keep it as minimal as possible, i.e., the total memory requirements for the output should equal the sum of the compacted sizes of the GASes.
    // Hence, it should be avoided to waste memory by allocating buffers that are bigger than what is required for a compacted GAS.
    //
    // The peak memory consumption effectively defines the efficiency of the algorithm.
    // If memory was unlimited, compaction isn't needed at all.
    // A lower bound for the peak memory consumption during the build is the output of the process, the size of the compacted GASes.
    // Peak memory consumption effectively defines the memory pool available during the batch building and compaction of GASes.
    //
    // The algorithm estimates the size of the compacted GASes by a give compaction ratio as well as the computed build size of each GAS.
    // The compaction ratio is defined as: size of compacted GAS / size of build output of GAS.
    // The validity of this estimate therefore depends on the assumed compaction ratio.
    // The current algorithm assumes a fixed compaction ratio.
    // Other strategies could be:
    // - update the compaction ration on the fly by do statistics on the already processed GASes to have a better guess for the remaining batches
    // - multiple compaction rations by type of GAS (e.g., motion vs static), since the type of GAS impacts the compaction ratio
    // Further, compaction may be skipped for GASes that do not benefit from compaction (compaction ratio of 1.0).
    //
    // Before selecting GASes for a batch, all GASes are sorted by size (their build size).
    // Big GASes are handled before smaller GASes as this will increase the likelihood of the peak memory consumption staying close to the minimal memory consumption.
    // This also increase the benefit of batching since small GASes that benefit most from avoiding synchronizations are built "together".
    // The minimum batch size is one GAS to ensure forward process.
    //
    // Goal:
    // Estimate the required output size (the minimal peak memory consumption) and work within these bounds.
    // Batch process GASes as long as they are expected to fit into the memory bounds (non strict).
    //
    // Assumptions:
    // The inputs to each GAS are already in device memory and are needed afterwards.
    // Otherwise this could be factored into the peak memory consumption.
    // E.g., by uploading the input data to the device only just before building the GAS and releasing it right afterwards.
    //
    // Further, the peak memory consumption of the application / system is influenced by many factors unknown to this algorithm.
    // E.g., if it is known that a big pool of memory is needed after GAS building anyways (e.g., textures that need to be present on the device),
    // peak memory consumption will be higher eventually and the GAS build process could already make use of a bigger memory pool.
    //
    // TODOs:
    // - compaction ratio estimation / updating
    // - handling of non-compactable GASes
    // - integration of GAS input data upload / freeing
    // - add optional hard limits / check for hard memory limits (shrink batch size / abort, ...)
    //////////////////////////////////////////////////////////////////////////

    // Magic constants:

    // see explanation above
    constexpr double initialCompactionRatio = 0.5;

    // It is assumed that trace is called later when the GASes are still in memory.
    // We know that the memory consumption at that time will at least be the compacted GASes + some CUDA stack space.
    // Add a "random" 250MB that we can use here, roughly matching CUDA stack space requirements.
    constexpr size_t additionalAvailableMemory = 250 * 1024 * 1024;

    //////////////////////////////////////////////////////////////////////////

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    struct GASInfo {
        std::vector<OptixBuildInput> buildInputs;
        OptixAccelBufferSizes gas_buffer_sizes;
        std::shared_ptr<MeshGroup> mesh;
    };
    std::multimap<size_t, GASInfo> gases;
    size_t totalTempOutputSize = 0;

    for(size_t i=0; i<m_meshes.size(); ++i) {
        auto& mesh = m_meshes[i];

        const size_t num_subMeshes =  mesh->indices.size();
        std::vector<OptixBuildInput> buildInputs(num_subMeshes);

        assert(mesh->positions.size() == num_subMeshes &&
            mesh->normals.size()   == num_subMeshes &&
            mesh->texcoords.size() == num_subMeshes);

        for(size_t i = 0; i < num_subMeshes; ++i) {
            OptixBuildInput& triangle_input                          = buildInputs[i];
            memset(&triangle_input, 0, sizeof(OptixBuildInput));
            triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes         =
                mesh->positions[i].byte_stride ?
                mesh->positions[i].byte_stride :
                sizeof(float3),
                triangle_input.triangleArray.numVertices             = mesh->positions[i].count;
            triangle_input.triangleArray.vertexBuffers               = &(mesh->positions[i].data);
            triangle_input.triangleArray.indexFormat                 =
                mesh->indices[i].elmt_byte_size == 2 ?
                OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 :
                OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes          =
                mesh->indices[i].byte_stride ?
                mesh->indices[i].byte_stride :
                mesh->indices[i].elmt_byte_size*3;
            triangle_input.triangleArray.numIndexTriplets            = mesh->indices[i].count / 3;
            triangle_input.triangleArray.indexBuffer                 = mesh->indices[i].data;
            triangle_input.triangleArray.flags                       = &triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords               = 1;
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        optixCheck(optixAccelComputeMemoryUsage(
                context_.get(),
                &accel_options,
                buildInputs.data(),
                static_cast<unsigned int>(num_subMeshes),
                &gas_buffer_sizes));

        totalTempOutputSize += gas_buffer_sizes.outputSizeInBytes;
        GASInfo g = {std::move( buildInputs ), gas_buffer_sizes, mesh};
        gases.emplace( gas_buffer_sizes.outputSizeInBytes, g );
    }

    size_t totalTempOutputProcessedSize = 0;
    size_t usedCompactedOutputSize = 0;
    double compactionRatio = initialCompactionRatio;

    OptixBuffer<char> d_temp;
    OptixBuffer<char> d_temp_output;
    OptixBuffer<size_t> d_temp_compactedSizes;

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    while(!gases.empty()) {
        // The estimated total output size that we end up with when using compaction.
        // It defines the minimum peak memory consumption, but is unknown before actually building all GASes.
        // Working only within these memory constraints results in an actual peak memory consumption that is very close to the minimal peak memory consumption.
        size_t remainingEstimatedTotalOutputSize = static_cast<size_t>((totalTempOutputSize - totalTempOutputProcessedSize) * compactionRatio);
        size_t availableMemPoolSize = remainingEstimatedTotalOutputSize + additionalAvailableMemory;
        // We need to fit the following things into availableMemPoolSize:
        // - temporary buffer for building a GAS (only during build, can be cleared before compaction)
        // - build output buffer of a GAS
        // - size (actual number) of a compacted GAS as output of a build
        // - compacted GAS

        size_t batchNGASes                    = 0;
        size_t batchBuildOutputRequirement    = 0;
        size_t batchBuildMaxTempRequirement   = 0;
        size_t batchBuildCompactedRequirement = 0;
        for(auto &gas : gases) {
            batchBuildOutputRequirement += gas.second.gas_buffer_sizes.outputSizeInBytes;
            batchBuildCompactedRequirement += static_cast<size_t>(gas.second.gas_buffer_sizes.outputSizeInBytes * compactionRatio);
            // roughly account for the storage of the compacted size, although that goes into a separate buffer
            batchBuildOutputRequirement += 8ull;
            // make sure that all further output pointers are 256 byte aligned
            batchBuildOutputRequirement = roundUp<size_t>( batchBuildOutputRequirement, 256ull );
            // temp buffer is shared for all builds in the batch
            batchBuildMaxTempRequirement = std::max( batchBuildMaxTempRequirement, gas.second.gas_buffer_sizes.tempSizeInBytes );
            batchNGASes++;
            if((batchBuildOutputRequirement + batchBuildMaxTempRequirement + batchBuildCompactedRequirement) > availableMemPoolSize) {
                break;
            }
        }

        d_temp.lazyAllocate(batchBuildMaxTempRequirement, 1.1);
        d_temp_output.lazyAllocate(batchBuildOutputRequirement, 1.1);
        d_temp_compactedSizes.lazyAllocate(batchNGASes, 2, 20);

        // sum of size of compacted GASes
        size_t batchCompactedSize = 0;

        auto it = gases.rbegin();
        for(size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i, it++) {
            emitProperty.result = d_temp_compactedSizes.get(i);
            GASInfo& info = it->second;

            optixCheck(optixAccelBuild(
                    context_.get(),
                    nullptr,   // CUDA stream
                    &accel_options,
                    info.buildInputs.data(),
                    static_cast<unsigned int>( info.buildInputs.size() ),
                    d_temp.get(),
                    d_temp.byteSize(),
                    d_temp_output.get( tempOutputAlignmentOffset ),
                    info.gas_buffer_sizes.outputSizeInBytes,
                    &info.mesh->gas_handle,
                    &emitProperty,  // emitted property list
                    1               // num emitted properties
                    ));

            tempOutputAlignmentOffset += roundUp<size_t>( info.gas_buffer_sizes.outputSizeInBytes, 256ull );
        }

        // trash d_temp if it is at least 20MB in size
        if(d_temp.byteSize() > (20 << 20)) {
            d_temp.clear();
        }

        // download all compacted sizes to allocate final output buffers for these GASes
        std::vector<size_t> h_compactedSizes(batchNGASes);
        d_temp_compactedSizes.copyToHost(h_compactedSizes.data());

        //////////////////////////////////////////////////////////////////////////
        // TODO:
        // Now we know the actual memory requirement of the compacted GASes.
        // Based on that we could shrink the batch if the compaction ratio is bad and we need to strictly fit into the/any available memory pool.
        bool canCompact = false;
        it = gases.rbegin();
        for(size_t i = 0; i < batchNGASes; ++i, it++) {
            GASInfo& info = it->second;
            if(info.gas_buffer_sizes.outputSizeInBytes > h_compactedSizes[i]) {
                canCompact = true;
                break;
            }
        }

        if(canCompact) {
            //////////////////////////////////////////////////////////////////////////
            // "batch allocate" the compacted buffers
            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i ) {
                GASInfo& info = it->second;
                batchCompactedSize += h_compactedSizes[i];
                info.mesh->d_gas_output.allocate(h_compactedSizes[i]);
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;
                it++;
            }

            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                optixCheck(optixAccelCompact(
                        context_.get(),
                        nullptr,
                        info.mesh->gas_handle,
                        reinterpret_cast<CUdeviceptr>(info.mesh->d_gas_output.get()),
                        h_compactedSizes[i],
                        &info.mesh->gas_handle
                        ));
                it++;
            }
        } else {
            it = gases.rbegin();
            for(size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i) {
                GASInfo& info = it->second;
                info.mesh->d_gas_output = d_temp_output.get( tempOutputAlignmentOffset );
                batchCompactedSize += h_compactedSizes[i];
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;

                tempOutputAlignmentOffset += roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
                it++;
            }
            d_temp_output.release(); // TODO проверить, что тут точно нет утечки
        }

        usedCompactedOutputSize += batchCompactedSize;

        gases.erase( it.base(), gases.end() );
    }
}


///TODO
struct Instance
{
    float transform[12];
};

void Scene::buildInstanceAccel( int rayTypeCount )
{
    const size_t num_instances = m_meshes.size();

    std::vector<OptixInstance> optix_instances( num_instances );

    unsigned int sbt_offset = 0;
    for( size_t i = 0; i < m_meshes.size(); ++i )
    {
        auto  mesh = m_meshes[i];
        auto& optix_instance = optix_instances[i];
        memset( &optix_instance, 0, sizeof( OptixInstance ) );

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = static_cast<unsigned int>( i );
        optix_instance.sbtOffset         = sbt_offset;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = mesh->gas_handle;
        memcpy( optix_instance.transform, mesh->transform.getData(), sizeof( float ) * 12 );

        sbt_offset += static_cast<unsigned int>( mesh->indices.size() ) * rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
    }

    OptixBuffer<OptixInstance> d_instances(num_instances);
    d_instances.copyFromHost(optix_instances.data());

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances.get();
    instance_input.instanceArray.numInstances = static_cast<unsigned int>( num_instances );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    optixCheck(optixAccelComputeMemoryUsage(
                context_.get(),
                &accel_options,
                &instance_input,
                1, // num build inputs
                &ias_buffer_sizes
                ));

    CudaDeviceBuffer d_temp_buffer(ias_buffer_sizes.tempSizeInBytes);
    d_ias_output_buffer_.allocate(ias_buffer_sizes.outputSizeInBytes);

    optixCheck(optixAccelBuild(
                context_.get(),
                nullptr,                  // CUDA stream
                &accel_options,
                &instance_input,
                1,                  // num build inputs
                reinterpret_cast<CUdeviceptr>(d_temp_buffer.get()),
                ias_buffer_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(d_ias_output_buffer_.get()),
                ias_buffer_sizes.outputSizeInBytes,
                &m_ias_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ));
}

void Scene::rebuildInstanceAccelAsync(cudaStream_t stream, int rayTypeCount)
{
    const size_t num_instances = m_meshes.size();

    std::vector<OptixInstance> optix_instances(num_instances);

    unsigned int sbt_offset = 0;
    for (size_t i = 0; i < m_meshes.size(); ++i) {
        auto mesh = m_meshes[i];
        auto &optix_instance = optix_instances[i];
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId = static_cast<unsigned int>( i );
        optix_instance.sbtOffset = sbt_offset;
        optix_instance.visibilityMask = 1;
        optix_instance.traversableHandle = mesh->gas_handle;
        Matrix4x4 m = mesh->transform;
        memcpy(optix_instance.transform, m.getData(), sizeof(float) * 12);

        sbt_offset += static_cast<unsigned int>( mesh->indices.size())
                * rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
    }

    if (d_instances_.size() < num_instances) {
        d_instances_.allocate(num_instances);
    }
    d_instances_.copyFromHost(optix_instances.data(), stream);

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances_.get();
    instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    optixCheck(optixAccelComputeMemoryUsage(
            context_.get(),
            &accel_options,
            &instance_input,
            1, // num build inputs
            &ias_buffer_sizes
    ));

    if(d_temp_buffer_.size() < ias_buffer_sizes.tempSizeInBytes) {
        d_temp_buffer_.allocate(ias_buffer_sizes.tempSizeInBytes);
    }
    if(d_ias_output_buffer_.size() < ias_buffer_sizes.outputSizeInBytes) {
        d_ias_output_buffer_.allocate(ias_buffer_sizes.outputSizeInBytes);
    }
    optixCheck(optixAccelBuild(
            context_.get(),
            stream, // CUDA stream
            &accel_options,
            &instance_input,
            1, // num build inputs
            reinterpret_cast<CUdeviceptr>(d_temp_buffer_.get()),
            ias_buffer_sizes.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(d_ias_output_buffer_.get()),
            ias_buffer_sizes.outputSizeInBytes,
            &m_ias_handle,
            nullptr, // emitted property list
            0 // num emitted properties
    ) );
}


void Scene::createPTXModule(std::string ptx)
{

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur            = false;
    m_pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues          = whitted::NUM_PAYLOAD_VALUES;
    m_pipeline_compile_options.numAttributeValues        = 2; // TODO
    m_pipeline_compile_options.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    m_ptx_module  = {};
    optixCheckLog(std::bind(optixModuleCreateFromPTX,
                context_.get(),
                &module_compile_options,
                &m_pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                std::placeholders::_1,
                std::placeholders::_2,
                &m_ptx_module
                ));
}


void Scene::createProgramGroups()
{
    OptixProgramGroupOptions program_group_options = {};

    //
    // Ray generation
    //
    {

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

        optixCheckLog(std::bind(optixProgramGroupCreate,
                    context_.get(),
                    &raygen_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    &m_raygen_prog_group
                    )
                );
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = m_ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_radiance";
        optixCheckLog(std::bind(optixProgramGroupCreate,
                    context_.get(),
                    &miss_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    &m_radiance_miss_group
                    )
                );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        optixCheckLog(std::bind(optixProgramGroupCreate,
                    context_.get(),
                    &miss_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    &m_occlusion_miss_group
                    )
                );
    }

    //
    // Hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        optixCheckLog(std::bind(optixProgramGroupCreate,
                    context_.get(),
                    &hit_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    &m_radiance_hit_group
                    )
                );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        optixCheckLog(std::bind(optixProgramGroupCreate,
                    context_.get(),
                    &hit_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    &m_occlusion_hit_group
                    ));
    }
}

void Scene::createPipeline()
{
    OptixProgramGroup program_groups[] =
    {
        m_raygen_prog_group,
        m_radiance_miss_group,
        m_occlusion_miss_group,
        m_radiance_hit_group,
        m_occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 2;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
//    pipeline_link_options.overrideUsesMotionBlur = false;

    optixCheckLog(std::bind(optixPipelineCreate,
                context_.get(),
                &m_pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                std::placeholders::_1,
                std::placeholders::_2,
                &m_pipeline
                ));
}


void Scene::createSBT()
{
    sbt_.allocateRecords<EmptyRecord, EmptyRecord>(
            m_raygen_prog_group,
            { m_radiance_miss_group, m_occlusion_miss_group }
            );

    std::vector<HitGroupRecord> hitgroup_records;
    for(const auto &mesh : m_meshes) {
        for(size_t i = 0; i < mesh->material_idx.size(); ++i) {
            HitGroupRecord rec = {};
            optixCheck(optixSbtRecordPackHeader( m_radiance_hit_group, &rec ));
            rec.data.geometry_data.type                    = GeometryData::TRIANGLE_MESH;
            rec.data.geometry_data.triangle_mesh.positions = mesh->positions[i];
            rec.data.geometry_data.triangle_mesh.normals   = mesh->normals[i];
            rec.data.geometry_data.triangle_mesh.texcoords = mesh->texcoords[i];
            rec.data.geometry_data.triangle_mesh.indices   = mesh->indices[i];

            const unsigned mat_idx  = mesh->material_idx[i];
            if( mat_idx >= 0 )
                rec.data.material_data = m_materials[mat_idx];
            else
                rec.data.material_data = MaterialData();
            hitgroup_records.push_back( rec );

            optixCheck( optixSbtRecordPackHeader( m_occlusion_hit_group, &rec ) );
            hitgroup_records.push_back( rec );
        }
    }
    sbt_.allocateHitGroups(hitgroup_records);
}

std::shared_ptr<CudaTextureObject> sutil::loadFileToTex(const std::string& fn)
{
    std::string err;
    std::vector<unsigned char> bytes;
    if(!tinygltf::ReadWholeFile(&bytes, &err, fn, nullptr)) {
        LOGC(ERROR) << "Can't read " << fn << ": " << err << std::endl;
        return nullptr;
    }

    tinygltf::Image image ;
    std::string warn;
    if(!tinygltf::LoadImageData(&image, 0, &err, &warn, 0, 0, bytes.data(), bytes.size(), nullptr)) {
        LOGC(ERROR) << "Can't parse " << fn << ": " << err << std::endl;
        return nullptr;
    }

    int32_t pitch;
    CudaArray cuda_image;
    if(image.bits == 8) {
        pitch = image.width * image.component * sizeof(uint8_t);
        cuda_image.allocate<uchar4>(image.width, image.height);
    } else if(image.bits == 16) {
        pitch = image.width * image.component * sizeof(uint16_t);
        cuda_image.allocate<ushort4>(image.width, image.height);
    } else {
        LOGC(ERROR) << "Unsupported bits/component in image: " << fn << std::endl;
        return nullptr;
    }
    cuda_image.copyFromHost(image.image.data(), pitch);

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0; // TODO: glTF assumes sRGB for base_color -- handle in shader

    // Create texture object
    return std::make_shared<CudaTextureObjectImage>(std::move(cuda_image), &tex_desc);
}


void Scene::calcNewTransforms(const shared_ptr<SceneNode> &node)
{
    const Matrix4x4 &parent_matrix = (node->parent < 0) ? Matrix4x4::identity() : nodes_[node->parent]->transform;
    auto& gltf_node = model_.nodes[node->id];
    Matrix4x4 translation = gltf_node.translation.empty() ?
            Matrix4x4::identity() :
            Matrix4x4::translate(make_float3(gltf_node.translation));

    Matrix4x4 rotation = gltf_node.rotation.empty() ?
            Matrix4x4::identity() :
            Quaternion(make_float4(gltf_node.rotation)).rotationMatrix();

    Matrix4x4 scale = gltf_node.scale.empty() ?
            Matrix4x4::identity() :
            Matrix4x4::scale(make_float3(gltf_node.scale));

    std::vector<float> gltf_matrix;
    for(double x : gltf_node.matrix) {
        gltf_matrix.push_back(static_cast<float>(x));
    }
    const Matrix4x4 matrix = gltf_node.matrix.empty() ?
            Matrix4x4::identity() :
            Matrix4x4(reinterpret_cast<float*>(gltf_matrix.data())).transpose();

    auto range = animations_.equal_range(node->id);
    for(auto it = range.first; it != range.second; ++it) {
        if(it->second->hasTranslate(node->id)) {
            translation = it->second->translate(node->id, last_time_);
        } else if(it->second->hasRotation(node->id)) {
            rotation = it->second->rotation(node->id, last_time_);
        } else if(it->second->hasScale(node->id)) {
            rotation = it->second->scale(node->id, last_time_);
        }
    }

    const Matrix4x4 node_xform = parent_matrix * matrix * translation * rotation * scale;
    node->transform = node_xform;

    if( gltf_node.mesh != -1 ) {
        auto ptrs = mesh_by_node.equal_range(gltf_node.name);
        for( auto ptr = ptrs.first ; ptr!= ptrs.second; ptr++) {
            std::shared_ptr<sutil::Scene::MeshGroup> mesh = ptr->second;
            mesh->transform = node_xform;
            mesh->world_aabb.transform(node_xform); // TODO убедиться что правильно пересчитывает.
        }
    }
    if(gltf_node.camera != -1) {
        cameras_[gltf_node.camera]->setTransform(node_xform);
    }
    for(int32_t child : gltf_node.children) {
        calcNewTransforms(nodes_.at(child));
    }
}

void Scene::addCamera(int node_id)
{
    const auto& gltf_node = model_.nodes[node_id];
    const auto& gltf_camera = model_.cameras[gltf_node.camera];
    LOGC(DEBUG) << "Processing camera '" << gltf_camera.name << "'\n" << "\ttype: " << gltf_camera.type << "\n";
    if( gltf_camera.type != "perspective" ) {
        LOGC(WARNING) << "\tskipping non-perpective camera\n";
        return;
    }

    addCamera(make_shared<Camera>(gltf_camera.name,
                                  rad2deg(gltf_camera.perspective.yfov),
                                  gltf_camera.perspective.aspectRatio));
}

void Scene::addMesh(int node_id)
{
    const auto& gltf_node = model_.nodes[node_id];
    const auto& gltf_mesh = model_.meshes[gltf_node.mesh];
    LOGC(DEBUG) << "Processing glTF mesh: '" << gltf_mesh.name << "'\n";
    LOGC(DEBUG) << "\tNum mesh primitive groups: " << gltf_mesh.primitives.size() << std::endl;
    for( auto& gltf_primitive : gltf_mesh.primitives ) {
        if( gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES ) { // Ignore non-triangle meshes
            LOGC(DEBUG) << "\tNon-triangle primitive: skipping\n";
            continue;
        }

        auto mesh = std::make_shared<Scene::MeshGroup>();
        addMesh( mesh );
        mesh_by_node.insert(std::make_pair(gltf_node.name, mesh));

        mesh->name = gltf_mesh.name;
        mesh->indices.push_back( bufferViewFromGLTF<uint32_t>(gltf_primitive.indices));
        mesh->material_idx.push_back( gltf_primitive.material );
        LOGC(DEBUG) << "\t\tNum triangles: " << mesh->indices.back().count / 3 << std::endl;

        assert( gltf_primitive.attributes.find( "POSITION" ) !=  gltf_primitive.attributes.end() );
        const int32_t pos_accessor_idx =  gltf_primitive.attributes.at( "POSITION" );
        mesh->positions.push_back(bufferViewFromGLTF<float3>(pos_accessor_idx));

        const auto& pos_gltf_accessor = model_.accessors[ pos_accessor_idx ];
        mesh->object_aabb = Aabb(
            make_float3_from_double(
                pos_gltf_accessor.minValues[0],
                pos_gltf_accessor.minValues[1],
                pos_gltf_accessor.minValues[2]
                ),
            make_float3_from_double(
                pos_gltf_accessor.maxValues[0],
                pos_gltf_accessor.maxValues[1],
                pos_gltf_accessor.maxValues[2]
                ) );
        mesh->world_aabb = mesh->object_aabb;

        auto normal_accessor_iter = gltf_primitive.attributes.find( "NORMAL" ) ;
        if( normal_accessor_iter  !=  gltf_primitive.attributes.end() ) {
            LOGC(DEBUG) << "\t\tHas vertex normals: true\n";
            mesh->normals.push_back( bufferViewFromGLTF<float3>(normal_accessor_iter->second));
        } else {
            LOGC(DEBUG) << "\t\tHas vertex normals: false\n";
            mesh->normals.push_back( bufferViewFromGLTF<float3>(-1));
        }

        auto texcoord_accessor_iter = gltf_primitive.attributes.find( "TEXCOORD_0" ) ;
        if( texcoord_accessor_iter  !=  gltf_primitive.attributes.end() ) {
            LOGC(DEBUG) << "\t\tHas texcoords: true\n";
            mesh->texcoords.push_back( bufferViewFromGLTF<float2>(texcoord_accessor_iter->second));
        } else {
            LOGC(DEBUG) << "\t\tHas texcoords: false\n";
            mesh->texcoords.push_back( bufferViewFromGLTF<float2>(-1));
        }
    }
}

void Scene::updateAnimated()
{
    animated_nodes_.clear();
    for(auto& node : root_nodes_) {
        findAnimated(node);
    }
}

void Scene::findAnimated(int id)
{
    if(animations_.find(id) != animations_.end()) {
        animated_nodes_.push_back(id);
        return;
    }
    for(auto& child : model_.nodes[id].children) {
        findAnimated(child);
    }
}

void Scene::doAnimation(uint64_t time_now_us)
{
    last_time_ = (time_now_us % max_animation_time_) * 1e-6;
    for(auto& node : animated_nodes_) {
        calcNewTransforms(nodes_[node]);
    }
}
