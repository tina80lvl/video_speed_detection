//
// Created by dipet on 08.04.2020.
//

#ifndef PHOTORADARSOURCES_OPTIX_UTILS_H
#define PHOTORADARSOURCES_OPTIX_UTILS_H

#include <Log.h>

#include "cuda/cuda_utils.h"

#include <optix.h>
#include <optix_stubs.h>

#include <iomanip>
#include <iostream>
#include <functional>

inline void optixCheck(OptixResult ret)
{
	if (ret != OPTIX_SUCCESS) {
		std::cerr << "Optix failure: " << optixGetErrorString(ret) << std::endl;
		exit(1);
	}
}

inline void optixCheckLog(const std::function<OptixResult(char*, size_t*)>& func)
{
	char log[2048];
	size_t size = sizeof(log);
	OptixResult ret = func(log, &size);
	if (ret != OPTIX_SUCCESS) {
		std::cerr << "Optix failure: " << optixGetErrorString(ret) << std::endl;
		std::cerr << "Log: " << ( size > sizeof(log) ? "<TRUNCATED> " : "") << log << std::endl;
		exit(1);
	}
}

template <typename T>
class OptixBuffer
{
public:
	OptixBuffer(size_t size = 0)
	{
		if(size)
			allocate(size);
	}
	~OptixBuffer()
	{
		clear();
	}

	void allocate(size_t size)
	{
		cudaCheck(cudaFree(ptr_));
		cudaCheck(cudaMalloc(&ptr_, size * sizeof(T)));
		size_ = size;
		allocated_ = size;
	}

	// size - новый размер.
	// ratio - отношение после которого делать ресайз.
	// max_size - размер (в МБ), по достижении которого будет ресайз.
	void lazyAllocate(size_t size, double ratio, size_t max_size = 0)
	{
		if(size > allocated_ || (static_cast<double>(allocated_) / size > ratio && allocated_ > (max_size  << 20))) {
			allocate(size);
		} else {
			size_ = size;
		}
	}

	CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>(ptr_); }
	CUdeviceptr get(size_t index) const { return reinterpret_cast<CUdeviceptr>(ptr_ + index); }
	void clear()
	{
		cudaCheck(cudaFree(ptr_));
		size_ = 0;
		allocated_ = 0;
		ptr_ = nullptr;
	}

	inline void copyFromHost(const T* data, const CudaStreamWrapper& stream)
	{
		copyFromHost(data, stream.get());
	}

	inline void copyFromHost(const T* data, cudaStream_t stream = nullptr)
	{
		cudaCheck(cudaMemcpyAsync(ptr_, data, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
	}

	inline void copyToHost(T* data, const CudaStreamWrapper& stream) const
	{
		copyToHost(data, stream.get());
	}

	inline void copyToHost(T* data, cudaStream_t stream = nullptr) const
	{
		cudaCheck(cudaMemcpyAsync(data, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}

	inline size_t size() const { return size_; }
	inline size_t byteSize() const { return size_ * sizeof(T); }

	T* release()
	{
		allocated_ = 0;
		size_ = 0;
		T* current = ptr_;
		ptr_ = nullptr;
		return current;
	}

private:
	size_t size_ = 0;
	size_t allocated_ = 0;
	T* ptr_ = nullptr;
};

class OptixContext
{
public:
	OptixContext() = default;
	explicit OptixContext(
			CUcontext cuda_context,
			OptixLogCallback log_function = &OptixContext::default_log,
			int log_level = 4,
			void* log_data = nullptr
			)
	{
		create(cuda_context, log_function, log_level, log_data);
	}
	~OptixContext()
	{
		optixCheck(optixDeviceContextDestroy(context_));
	}

	void create(
			CUcontext cuda_context = nullptr,
			OptixLogCallback log_function = &OptixContext::default_log,
			int log_level = 4,
			void* log_data = nullptr
			)
	{
		if(context_) {
			optixCheck(optixDeviceContextDestroy(context_));
		}
		optixCheck(optixInit());
		OptixDeviceContextOptions options;
		options.logCallbackFunction = log_function;
		options.logCallbackLevel = log_level;
		options.logCallbackData = log_data;
		optixCheck(optixDeviceContextCreate(cuda_context, &options, &context_));
	}

	inline OptixDeviceContext get() const { return context_; }

	template<typename T = unsigned>
	inline T getProperty(OptixDeviceProperty property) const
	{
		T val;
		optixCheck(optixDeviceContextGetProperty(context_, property, &val, sizeof(T)));
		return val;
	}

	inline void setLogCallback(OptixLogCallback callback, unsigned level, void* data)
	{
		optixCheck(optixDeviceContextSetLogCallback(context_, callback, data, level));
	}

	inline void setCacheEnabled(bool flag)
	{
		optixCheck(optixDeviceContextSetCacheEnabled(context_, flag ? 1 : 0));
	}

	inline void setCacheLocation(const std::string& location)
	{
		optixCheck(optixDeviceContextSetCacheLocation(context_, location.c_str()));
	}

	inline void setCacheDatabaseSizes(size_t low, size_t high)
	{
		optixCheck(optixDeviceContextSetCacheDatabaseSizes(context_, low, high));
	}

	inline bool getCacheEnabled() const
	{
		int res;
		optixCheck(optixDeviceContextGetCacheEnabled(context_, &res));
		return static_cast<bool>(res);
	}

	inline std::string getCacheLocation() const
	{
		const size_t length = 4096;
		char location[length];
		optixCheck(optixDeviceContextGetCacheLocation(context_, location, length));
		return location;
	}

	std::pair<size_t, size_t> getCacheDatabaseSizes()
	{
		size_t low, high;
		optixCheck(optixDeviceContextGetCacheDatabaseSizes(context_, &low, &high));
		return { low, high };
	}

public:
	static void default_log( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
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

private:
	OptixDeviceContext context_ = nullptr;
};

#endif //PHOTORADARSOURCES_OPTIX_UTILS_H
