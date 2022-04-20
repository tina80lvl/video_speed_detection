//
// Created by bond on 18.11.2019.
//

#ifndef PHOTORADARSOURCES_MESHREADER_H
#define PHOTORADARSOURCES_MESHREADER_H

#include "CMem.h"
#include "Pollable.h"
#include "HeavyDataReader.h"
#include "FileDescriptorPoller.h"
#include "TimeoutDispatcher.h"

#include "utils/gplay/cuda/cuda_utils.h"

#include "sutil/Scene.h"
#include "sutil/Matrix.h"

#include "sutil/Camera.h"

#include "cuda/whitted.h"

#include <string>

#include <optix.h>

struct sc_mem_res;
struct compilation
{
	// in
	unsigned num_awaited_depths;
	sc_mem_res *img_mem;
	// out (filled after render)
	std::vector<sc_mem_res *> depths;
	bool img_done;
};

/* ресурсы которые моут понадобится 1-й задаче в GPU*/
struct sc_mem_res
{
	CudaStreamWrapper stream; // stream for communicaion with GPU
	CudaEventWrapper finish_event; // event "task done"
	CudaDeviceBuffer params_cumem; // mem(in GPU DDR) for task params

	// task params (in CPU mem)
	uint64_t time_us = 0;
	float camera_fov_y = 0;
	unsigned launch_width = 0;  // dimensions image or depths
	unsigned launch_height = 0;  // dimensions image or depths

	// is output is image
	CudaMirroredBuffer color_buf;

	// is output is depths
	CudaMirroredBuffer depth_buf; // GPU mem
	uint64_t depth_buf_bus = 0;

	// is output is translation
	CudaMirroredBuffer trans_buf;
	uint64_t trans_buf_bus = 0;

	// used for compile 1 img +20 depth tasks together
	struct compilation *compilation = nullptr;
};

class MeshReader;
class MeshReaderGrabbedFrameDescriptor: public GrabbedFrameDescriptor
{
public:
	MeshReaderGrabbedFrameDescriptor(MeshReader *father, timeval capturingtime,
			timespec capturingclock, sc_mem_res *mem);
	~MeshReaderGrabbedFrameDescriptor() override;

	unsigned char *getContents() override;

	bool isClaimAllowed() override;

	void setDepths(std::vector<sc_mem_res *> depths);

private:
	MeshReader *father;
	sc_mem_res *mem;
	std::vector<sc_mem_res *> depths;
};

class MeshReader: public HeavyDataReader, public TimeoutHandler
{
public:
	MeshReader(TimeoutDispatcher *timeout_dispatcher, double play_fps,
			CMem *cmem);
	~MeshReader() override;

	// own
	bool
	init(unsigned num_img_bufs, const std::string &ptx_file, unsigned img_width,
			unsigned img_height, unsigned depths_per_image,
			unsigned depth_width, unsigned depth_height);
	void setCamDirection(float3 dirWS);
	bool openFile(const std::string &infile);

	float3 getCameraRotation() const;
	void rotateCamEuler(float3 angles);

public: //HeavyDataReader
	bool nextFrame(bool is_needed_release) override;
	unsigned pendedFrames() override;

public: //TimeoutHandler interface
	void timeoutHappened(int timeout_id) override;

private:
	void renderDone(sc_mem_res *mem);

	friend class MeshReaderGrabbedFrameDescriptor;
	void reenqueMem(sc_mem_res *mem, const std::vector<sc_mem_res *> &depths);
	bool isClaimAllowed(int thr) override ;

	bool
	addMem(unsigned img_width, unsigned img_height, unsigned depths_per_image,
			unsigned depth_width, unsigned depth_height);

private:
	TimeoutDispatcher *timeout_dispatcher;
	double play_fps; //ToDo: rm ?
	CMem *cmem;
	float2 camera_img_dim;
	unsigned depth_maps_per_image;
	float2 camera_depth_dim;

	std::string ptx; // GPU program for raytrace
	std::shared_ptr<CudaTextureObject> env_tex_; // world(env) in GPU mem
	BufferView<Light::Point> lights_bv; // lights descr in GPU mem

	std::shared_ptr<sutil::Camera> camera_img_;
	std::shared_ptr<sutil::Camera> camera_depth_;

	std::queue<sc_mem_res *> res_pool; // not used resources
	std::queue<sc_mem_res *> run_q; // in gpu, we wait finish_event
	std::set<compilation *> compilations; // 1 compilation is: 1 img + 20depths

	//time;
	uint64_t time_beg_us;
	uint64_t time_now_us;

	float m3x4_scene_to_img[12]; // for DEPTH_TO_CAM_MAP
public:
	sutil::Scene scene;
};

#endif //PHOTORADARSOURCES_MESHREADER_H
