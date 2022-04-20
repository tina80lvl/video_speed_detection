#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <set>
#include <thread>
#include <iostream>
#include <cuda_runtime.h>

inline bool cudaCheck(cudaError_t ret, bool strict = true)
{
	if (ret != cudaSuccess) {
        std::cerr << "Cuda failure: " << cudaGetErrorString(ret) << std::endl;
        if(!strict) {
            return false;
        }
        exit(1);
	}
    return true;
}

inline void cudaSleep(void* sleep)
{
	std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(*static_cast<int*>(sleep)));
}

class CudaEventWrapper;
class CudaStreamWrapper
{
public:

	CudaStreamWrapper()
	{
		cudaCheck(cudaStreamCreate(&stream_));
	}
	CudaStreamWrapper(const CudaStreamWrapper&) = delete;
	CudaStreamWrapper& operator=(const CudaStreamWrapper&) = delete;
	CudaStreamWrapper(CudaStreamWrapper&&) = delete;
	CudaStreamWrapper& operator=(CudaStreamWrapper&&) = delete;

	~CudaStreamWrapper()
	{
		cudaCheck(cudaStreamDestroy(stream_));
	}

	inline cudaStream_t get() const { return stream_; }

	void wait(CudaEventWrapper& event);
	inline void sleep(int* ms) { cudaCheck(cudaLaunchHostFunc(stream_, cudaSleep, ms)); }

private:
	cudaStream_t stream_;
};

class CudaEventWrapper
{
public:

	explicit CudaEventWrapper(unsigned flags = cudaEventDisableTiming)
	{
		cudaCheck(cudaEventCreateWithFlags(&event_, flags));
	}

	CudaEventWrapper(const CudaEventWrapper&) = delete;
	CudaEventWrapper& operator=(const CudaEventWrapper&) = delete;
	CudaEventWrapper(CudaEventWrapper&&) = delete;
	CudaEventWrapper& operator=(CudaEventWrapper&&) = delete;

	~CudaEventWrapper()
	{
		cudaCheck(cudaEventDestroy(event_));
	}

	inline cudaEvent_t get() const { return event_; }

	inline void record(const CudaStreamWrapper& stream) { cudaCheck(cudaEventRecord(event_, stream.get())); }

	inline void synchronize() { cudaCheck(cudaEventSynchronize(event_)); }
	bool isReady()
	{
		cudaError_t res = cudaEventQuery(event_);
		if(res == cudaSuccess) {
			return true;
		}
		if(res != cudaErrorNotReady) {
			cudaCheck(res);
		}
		return false;
	}

private:
	cudaEvent_t event_;
};

inline void CudaStreamWrapper::wait(CudaEventWrapper &event)
{
	cudaCheck(cudaStreamWaitEvent(stream_, event.get(), 0));
}

template <typename Allocator, typename Deallocator>
class CudaBuffer
{
public:
	CudaBuffer() = default;
	CudaBuffer(size_t size)
	{
		Allocator()(&ptr_, size);
	}
	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer& operator=(const CudaBuffer&) = delete;

	CudaBuffer(CudaBuffer&& rhs)
	{
		reset(rhs.ptr_, rhs.size_);
		rhs.ptr_ = nullptr;
	}

	CudaBuffer& operator=(CudaBuffer&& rhs)
	{
		if (this != &rhs) {
			reset(rhs.ptr_, rhs.size_);
			rhs.ptr_ = nullptr;
		}
		return *this;
	}

	~CudaBuffer()
	{
		reset();
	}

    bool allocate(size_t size, bool strict = true)
	{
		if(size_ != size) {
            if(!reset()) {
                return false;
            }
            size_ = size;
            return Allocator()(&ptr_, size, strict);
		}
        return true;
	}

	inline size_t size() const { return size_; }
	inline void* get() const { return ptr_; }
    inline bool reset(bool strict = true) { return reset(nullptr, 0, strict); }

protected:
    bool reset(void* ptr, size_t size, bool strict = true)
    {
        bool s = Deallocator()(ptr_, strict);
		ptr_ = ptr;
		size_ = size;
        return s;
	}

protected:
	size_t size_ = 0;
	void *ptr_ = nullptr;
};

struct DeviceAllocator
{
    bool operator()(void** ptr, size_t size, bool strict = true) { return cudaCheck(cudaMalloc(ptr, size), strict); }
};

struct DeviceDeallocator
{
    bool operator()(void* ptr, bool strict = true) { return cudaCheck(cudaFree(ptr), strict); }
};

struct HostAllocator
{
    bool operator()(void** ptr, size_t size, bool strict = true) { return cudaCheck(cudaMallocHost(ptr, size), strict); }
};

struct HostDeallocator
{
    bool operator()(void* ptr, bool strict = true) { return cudaCheck(cudaFreeHost(ptr), strict); }
};

class CudaDeviceBuffer :public CudaBuffer<DeviceAllocator, DeviceDeallocator>
{
public:
	using CudaBuffer::CudaBuffer;

	inline void copyFromHost(const void* src, size_t size, cudaStream_t stream = nullptr)
	{
		cudaCheck(cudaMemcpyAsync(ptr_, src, size, cudaMemcpyHostToDevice, stream));
	}

	inline void copyFromHost(const void* src, size_t size, const CudaStreamWrapper &stream)
	{
		copyFromHost(src, size, stream.get());
	}

	inline void copyToHost(void* dst, size_t size, cudaStream_t stream = nullptr) const
	{
		cudaCheck(cudaMemcpyAsync(dst, ptr_, size, cudaMemcpyDeviceToHost, stream));
	}

	inline void copyToHost(void* dst, size_t size, const CudaStreamWrapper &stream) const
	{
		copyToHost(dst, size, stream.get());
	}

    inline void copyToHostSync(void* dst, size_t size) const
    {
        cudaCheck(cudaMemcpy(dst, ptr_, size, cudaMemcpyDeviceToHost));
    }

    inline void copyFromHostSync(const void* src, size_t size)
    {
        cudaCheck(cudaMemcpy(ptr_, src, size, cudaMemcpyHostToDevice));
    }
};

using CudaHostBuffer = CudaBuffer<HostAllocator, HostDeallocator>;

class CudaMirroredBuffer
{
public:
	CudaMirroredBuffer() = default;
	CudaMirroredBuffer(size_t size, void *host_ptr = nullptr)
	{
		allocate(size, host_ptr);
	}
	~CudaMirroredBuffer()
	{
		if(non_cuda_host_) {
            cudaCheck(cudaHostUnregister(host_ptr_));
		}
	}

    bool allocate(size_t size, void *host_ptr = nullptr, bool strict = true)
	{
		if(host_ptr) {
			host_ptr_ = host_ptr;
#if 0 // cudaHostRegister return  invalid argument
			// TODO возможно стоит как-то поуправлять флагами.
            if(!cudaCheck(cudaHostRegister(host_ptr, size, 0), strict)) {
                return false;
            }
			non_cuda_host_ = true;
#endif
        } else {
            if(!host_buffer_.allocate(size, strict)) {
                return false;
            }
			host_ptr_ = host_buffer_.get();
		}
        return device_buffer_.allocate(size, strict);
	}

	inline void* getDeviceBuffer() const { return device_buffer_.get(); }

	inline void* getHostBuffer() const { return host_ptr_; }

	inline void hostToDevice(const cudaStream_t stream = nullptr)
	{
		device_buffer_.copyFromHost(host_ptr_, size(), stream);
	}

	inline void hostToDevice(const CudaStreamWrapper& stream)
	{
		device_buffer_.copyFromHost(host_ptr_, size(), stream);
	}

	inline void deviceToHost(const cudaStream_t stream = nullptr)
	{
		device_buffer_.copyToHost(host_ptr_, device_buffer_.size(), stream);
	}

	inline void deviceToHost(const CudaStreamWrapper& stream)
	{
		device_buffer_.copyToHost(host_ptr_, device_buffer_.size(), stream);
	}

	inline size_t size() const { return device_buffer_.size(); }

private:
	bool non_cuda_host_ = false;
	void *host_ptr_ = nullptr;
	CudaHostBuffer host_buffer_;
	CudaDeviceBuffer device_buffer_;
};

class CudaArray
{
public:
	CudaArray() = default;
	template<typename T>
	CudaArray(size_t width = 0, size_t height = 0, unsigned flags = 0)
	{
		allocate<T>(width, height, flags);
	}
	CudaArray(const CudaArray&) = delete;
	CudaArray& operator=(const CudaArray&) = delete;

	CudaArray(CudaArray&& rhs)
	{
		reset(rhs.release(), rhs.width_, rhs.height_);
	}

	CudaArray& operator=(CudaArray&& rhs)
	{
		if (this != &rhs) {
			reset(rhs.release(), rhs.width_, rhs.height_);
		}
		return *this;
	}

	~CudaArray()
	{
		reset();
	}

	template<typename T>
	void allocate(size_t width, size_t height, unsigned flags = 0)
	{
		reset();
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaCheck(cudaMallocArray(&ptr_, &desc, width, height, flags));
		height_ = height;
		width_ = width;
	}

	inline size_t width() const { return width_; }
	inline size_t height() const { return height_; }
	inline cudaArray_t get() const { return ptr_; }
	inline void reset() { reset(nullptr, 0, 0); }

	cudaArray_t release()
	{
		cudaArray_t current = ptr_;
		height_ = 0;
		width_ = 0;
		ptr_ = nullptr;
		return current;
	}

	inline void copyToHost(void* dst, size_t pitch, const CudaStreamWrapper& stream) const
	{
		copyToHost(dst, pitch, stream.get());
	}
	inline void copyToHost(void* dst, size_t pitch, cudaStream_t stream = nullptr) const
	{
		cudaCheck(cudaMemcpy2DFromArrayAsync(dst, pitch, ptr_, 0, 0, pitch, height_, cudaMemcpyDeviceToHost, stream));
	}

	inline void copyFromHost(const void* src, size_t pitch, const CudaStreamWrapper& stream)
	{
		copyFromHost(src, pitch, stream.get());
	}
	inline void copyFromHost(const void* src, size_t pitch, cudaStream_t stream = nullptr)
	{
		cudaCheck(cudaMemcpy2DToArrayAsync(ptr_, 0, 0, src, pitch, pitch, height_, cudaMemcpyHostToDevice, stream));
	}

protected:
	void reset(cudaArray_t ptr, size_t width, size_t height)
	{
		cudaCheck(cudaFreeArray(ptr_));
		ptr_ = ptr;
		height_ = height;
		width_ = width;
	}

protected:
	size_t width_ = 0;
	size_t height_ = 0;
	cudaArray_t ptr_ = nullptr;
};

class CudaTextureObject
{
public:
	CudaTextureObject() = default;
	CudaTextureObject(
			const cudaResourceDesc* resource_desc,
			const cudaTextureDesc* texture_desc,
			const cudaResourceViewDesc* resource_view_desc = nullptr
			)
	{
		allocate(resource_desc, texture_desc, resource_view_desc);
	}
	CudaTextureObject(const CudaTextureObject&) = delete;
	CudaTextureObject& operator=(const CudaTextureObject&) = delete;

	CudaTextureObject(CudaTextureObject&& rhs)
	:
		texture_(rhs.texture_)
	{
		rhs.texture_ = 0;
	}

	CudaTextureObject& operator=(CudaTextureObject&& rhs)
	{
		if (this != &rhs) {
			clear();
			texture_ = rhs.texture_;
			rhs.texture_ = 0;
		}
		return *this;
	}

	virtual ~CudaTextureObject()
	{
		clear();
	}

	virtual void allocate(
			const cudaResourceDesc* resource_desc,
			const cudaTextureDesc* texture_desc,
			const cudaResourceViewDesc* resource_view_desc = nullptr
			)
	{
		clear();
		cudaCheck(cudaCreateTextureObject(&texture_, resource_desc, texture_desc, resource_view_desc));
	}

	virtual inline cudaTextureObject_t get() const { return texture_; }
	virtual inline void clear()
	{
		cudaCheck(cudaDestroyTextureObject(texture_));
		texture_ = 0;
	}

protected:
	cudaTextureObject_t texture_ = 0;
};

class CudaTextureObjectImage : public CudaTextureObject
{
public:
	using CudaTextureObject::CudaTextureObject;
	CudaTextureObjectImage(
			CudaArray&& image,
			const cudaTextureDesc* texture_desc,
			const cudaResourceViewDesc* resource_view_desc = nullptr
			)
	{
		allocate(std::move(image), texture_desc, resource_view_desc);
	}

	CudaTextureObjectImage(CudaTextureObjectImage&& rhs)
	{
		image_ = std::move(rhs.image_);
		texture_ = rhs.texture_;
		rhs.texture_ = 0;
	}

	CudaTextureObjectImage& operator=(CudaTextureObjectImage&& rhs)
	{
		if (this != &rhs) {
			clear();
			texture_ = rhs.texture_;
			image_ = std::move(rhs.image_);
			rhs.texture_ = 0;
		}
		return *this;
	}

	using CudaTextureObject::allocate;
	void allocate(
			CudaArray&& image,
			const cudaTextureDesc* texture_desc,
			const cudaResourceViewDesc* resource_view_desc = nullptr
	)
	{
		cudaResourceDesc res_desc = {};
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array  = image.get();
		allocate(&res_desc, texture_desc, resource_view_desc);

		// clear будет вызван в allocate родителя, по этому присваиваем после.
		image_ = std::move(image);
	}

	inline cudaArray_t getImage() const { return image_.get(); }

	inline void clear() override
	{
		CudaTextureObject::clear();
		image_.reset();
	}

protected:
	CudaArray image_;
};

#endif // CUDA_HELPERS_H
