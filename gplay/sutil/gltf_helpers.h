#ifndef GLTF_HELPERS_H
#define GLTF_HELPERS_H

#include "tinygltf/tiny_gltf.h"

#include <limits>
#include <vector_types.h>

#include <Log.h>

template<typename T>
struct TypeParseTraits
{
	static const std::string name;
};

inline size_t gltfComponentType2TypeSize(int type)
{
	switch (type) {
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
	case TINYGLTF_COMPONENT_TYPE_BYTE:
		return 1;
	case TINYGLTF_COMPONENT_TYPE_SHORT:
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
		return 2;
	case TINYGLTF_COMPONENT_TYPE_INT:
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
	case TINYGLTF_COMPONENT_TYPE_FLOAT:
		return 4;
	case TINYGLTF_COMPONENT_TYPE_DOUBLE:
		return 8;
	default:
		return 0;
	}
}

template<typename T>
inline bool gltfComponentTypeIsSame(int type)
{
	switch (type) {
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
		return std::is_same<T, uint8_t>::value;
	case TINYGLTF_COMPONENT_TYPE_BYTE:
		return std::is_same<T, int8_t>::value;
	case TINYGLTF_COMPONENT_TYPE_SHORT:
		return std::is_same<T, int16_t>::value;
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
		return std::is_same<T, uint16_t>::value;
	case TINYGLTF_COMPONENT_TYPE_INT:
		return std::is_same<T, int32_t>::value;
	case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
		return std::is_same<T, uint32_t>::value;
	case TINYGLTF_COMPONENT_TYPE_FLOAT:
		return std::is_same<T, float>::value;
	case TINYGLTF_COMPONENT_TYPE_DOUBLE:
		return std::is_same<T, double>::value;
	default:
		return false;
	}
}

template<typename T>
static bool gltfTypeIsSame(int type, int component_type)
{
	std::string type_name = TypeParseTraits<T>::name;

	int num_components;
	try {
		num_components = std::stoi(std::string(1, type_name.back()));
		type_name = type_name.substr(0, type_name.size() - 1);
	} catch (const std::invalid_argument& e) {
		num_components = 1;
	}
	if(num_components < 0 || num_components > 4) {
		LOGC(ERROR) << "Unsupported type: " << typeid(T).name() << std::endl;
		exit(1);
	}

	bool component_is_same = false;
	if(type_name == "uchar") {
		component_is_same = gltfComponentTypeIsSame<uint8_t>(component_type);
	} else if(type_name == "char") {
		component_is_same = gltfComponentTypeIsSame<int8_t>(component_type);
	} else if(type_name == "ushort") {
		component_is_same = gltfComponentTypeIsSame<uint16_t>(component_type);
	} else if(type_name == "short") {
		component_is_same = gltfComponentTypeIsSame<int16_t>(component_type);
	} else if(type_name == "uint") {
		component_is_same = gltfComponentTypeIsSame<uint32_t>(component_type);
	} else if(type_name == "int") {
		component_is_same = gltfComponentTypeIsSame<int32_t>(component_type);
	} else if(type_name == "float") {
		component_is_same = gltfComponentTypeIsSame<float>(component_type);
	} else if(type_name == "double") {
		component_is_same = gltfComponentTypeIsSame<double>(component_type);
	} else {
		LOGC(ERROR) << "Unsupported type: " << typeid(T).name() << std::endl;
		exit(1);
	}
	if(!component_is_same) {
		return false;
	}

	switch (num_components) {
	case 1:
		return type == TINYGLTF_TYPE_SCALAR;
	case 2:
		return type == TINYGLTF_TYPE_VEC2;
	case 3:
		return type == TINYGLTF_TYPE_VEC3;
	case 4:
		return type == TINYGLTF_TYPE_VEC4;
	default:
		return false;
	}
}

class GltfBuffer
{
	// TODO Sparse data
public:
	GltfBuffer(const tinygltf::Model& model, int accessor_id)
	{
		accessor_ = &model.accessors[accessor_id];
		view_ = &model.bufferViews[accessor_->bufferView];
		const auto& buffer = model.buffers[view_->buffer];

		data_ = buffer.data.data() + accessor_->byteOffset + view_->byteOffset;
	}

	inline const uint8_t* data() const { return data_; }
	inline size_t size() const { return accessor_->count; }
	inline size_t stride() const { return view_->byteStride; }

	template<typename T>
	inline bool componentTypeIsSame() const { return gltfComponentTypeIsSame<T>(accessor_->componentType); }

	inline bool isScalar() const { return accessor_->type == TINYGLTF_TYPE_SCALAR; }
	inline bool isVec2() const { return accessor_->type == TINYGLTF_TYPE_VEC2; }
	inline bool isVec3() const { return accessor_->type == TINYGLTF_TYPE_VEC3; }
	inline bool isVec4() const { return accessor_->type == TINYGLTF_TYPE_VEC4; }
	inline bool isMat2() const { return accessor_->type == TINYGLTF_TYPE_MAT2; }
	inline bool isMat3() const { return accessor_->type == TINYGLTF_TYPE_MAT3; }
	inline bool isMat4() const { return accessor_->type == TINYGLTF_TYPE_MAT4; }

	template<typename T>
	inline bool typeIsSame() const { return gltfTypeIsSame<T>(accessor_->type, accessor_->componentType); }

	template<typename T>
	T get(size_t i) const
	{
		const T* ptr = reinterpret_cast<const T*>(data_) + i;
		return *ptr;
	}

    template<typename T>
    std::pair<T, T> getMinMax() const
    {
        std::pair<T, T> result = { get<T>(0), get<T>(0) };
        for(size_t i = 1; i < size(); ++i) {
            T val = get<T>(i);
            result.first = std::min(result.first, val);
            result.second = std::max(result.second, val);
        }
        return result;
    }

	template<typename T>
	inline T first() const { return get<T>(0); }
	template<typename T>
	inline T last() const { return get<T>(size() - 1); }

private:
	const uint8_t* data_;
	const tinygltf::Accessor* accessor_;
	const tinygltf::BufferView* view_;
};

#endif // GLTF_HELPERS_H
