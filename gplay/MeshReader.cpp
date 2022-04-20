#include "MeshReader.h"
#include "PerfCounter.h"
#include "Socket.h"
#include <string>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <cuda/whitted.h>
#include <cuda/Light.h>
#include <sutil/Scene.h>
#include <sutil/vec_math.h>
#include <iostream>
#include <fstream>
#include <sutil/Quaternion.h>
#include "utils/gplay/cuda/cuda_utils.h"
#include <sys/time.h>

#include <chrono>

#include "math_utils.h"

using namespace std;

//========== MeshReaderGrabbedFrameDescriptor =====================================
MeshReaderGrabbedFrameDescriptor::MeshReaderGrabbedFrameDescriptor(
		MeshReader *father, timeval capturingtime, timespec capturingclock,
		sc_mem_res *mem):
		GrabbedFrameDescriptor(capturingtime, capturingclock,
				static_cast<int>(mem->launch_width),
				static_cast<int>(mem->launch_height),
				mem->launch_width * mem->launch_height * 2, "YUV422P"),
		father(father), mem(mem)
{
}

unsigned char *MeshReaderGrabbedFrameDescriptor::getContents()
{
	return (unsigned char *) mem->color_buf.getHostBuffer();
}

MeshReaderGrabbedFrameDescriptor::~MeshReaderGrabbedFrameDescriptor()
{
	father->reenqueMem(mem, depths);
}

bool MeshReaderGrabbedFrameDescriptor::isClaimAllowed()
{
	return father->isClaimAllowed(2);
}

void
MeshReaderGrabbedFrameDescriptor::setDepths(vector<sc_mem_res *> new_depths)
{
	depths = move(new_depths);
	if (depths.empty())
		return;
	uint64_t min_time_us = depths[0]->time_us;
	size_t min_idx = 0;
	for (size_t i = 0; i < depths.size(); i++) {
		params.setParamU64("DEPTH_MAP_ADDR_" + to_string(i),
				(uint64_t) depths[i]->depth_buf_bus);
		if (depths[i]->time_us < min_time_us) {
			min_time_us = depths[i]->time_us;
			min_idx = i;
		}
	}
	params.setParamU64("DEPTH_TO_CAM_MAP_ADDR_0",
			(intptr_t) depths[min_idx]->trans_buf_bus);

	// temporary hack, if we cant't render 20 depths (performance limit)
	// copy last depth to other
	if (depths.size() < 20) {
		for (unsigned i = depths.size(); i < 20; i++) {
			params.setParamU64("DEPTH_MAP_ADDR_" + to_string(i),
					(intptr_t) depths[depths.size() - 1]->depth_buf_bus);
		}
	}
}

//---
MeshReader::MeshReader(TimeoutDispatcher *timeout_dispatcher, double play_fps,
		CMem *cmem):
		HeavyDataReader(false, ""), timeout_dispatcher(timeout_dispatcher),
		cmem(cmem), play_fps(play_fps), depth_maps_per_image(0),
		camera_img_dim({ 0, 0 }), camera_depth_dim({ 0, 0 }), time_beg_us(0),
		time_now_us(0)
{
	memset(m3x4_scene_to_img, 0, sizeof(m3x4_scene_to_img));
	// play_format: only "YUV422P" now
}

void freeByCmem(CMem *cmem, void *hostmem, uint64_t buf_bus, unsigned buf_len)
{
	host_mem_resource mem_res;
	mem_res.map_addr = (unsigned char *) hostmem;
	mem_res.map_size = static_cast<int>(buf_len);

	int n = static_cast<int>(log2((buf_len + 4095) / 4096));
	int order = static_cast<int>(log2((buf_len + 4095) / 4096 + pow(2, n) - 1));
	mem_res.cbuf.order = order;
	mem_res.cbuf.addrbus = buf_bus;
	mem_res.cbuf.addrvirt = 0;
	cmem->releaseHostRes(mem_res);
}

MeshReader::~MeshReader()
{
	cudaCheck(cudaFree((void *) lights_bv.data));

	// free unused resources,
	// only out_clmem,out_data_host_ptr must be cleaned
	while (!res_pool.empty()) {
		sc_mem_res *mem = res_pool.front();
		res_pool.pop();

		if (cmem) {
			freeByCmem(cmem, mem->trans_buf.getHostBuffer(), mem->trans_buf_bus,
					mem->trans_buf.size());
			freeByCmem(cmem, mem->depth_buf.getHostBuffer(), mem->depth_buf_bus,
					mem->depth_buf.size());
		}

		delete mem;
	}

	// free used resources
	while (!run_q.empty()) {
		sc_mem_res *mem = run_q.front();
		run_q.pop();
		mem->finish_event.synchronize();

		if (cmem) {
			freeByCmem(cmem, mem->trans_buf.getHostBuffer(), mem->trans_buf_bus,
					mem->trans_buf.size());
			freeByCmem(cmem, mem->depth_buf.getHostBuffer(), mem->depth_buf_bus,
					mem->depth_buf.size());
		}

		delete mem;
	}
}

string loadPtxFile(const string &filename)
{
	ifstream file;
	file.open(filename);
	if (!file) {
		LOGC(ERROR) << "can't open " << filename << endl;
		return "";
	}
	stringstream source_buffer;
	source_buffer << file.rdbuf();
	return source_buffer.str();
}

bool MeshReader::addMem(unsigned img_width, unsigned img_height,
		unsigned depths_per_image, unsigned depth_width, unsigned depth_height)
{
	sc_mem_res *mem = new sc_mem_res;

	unsigned n_pixels = img_width * img_height;
	if (!mem->color_buf.allocate(n_pixels * 2, nullptr, false)) { //YUV422P
		return false;
	}
	if (!mem->params_cumem.allocate(sizeof(whitted::LaunchParams), false)) {
		return false;
	}

	if (depths_per_image == 0) {
		res_pool.push(mem);
		return true;
	}

	//bufs for depths/translate
	unsigned n_pixels_depth = depth_width * depth_height;
	auto depth_buf_len = n_pixels_depth * sizeof(uint16_t);
	if (cmem) {
		host_mem_resource res = cmem->claimHostRes2(depth_buf_len);
		if (!res.map_size) {
			return false;
		}
		if (!mem->depth_buf.allocate(depth_buf_len, res.virtualAddr(), false)) {
			return false;
		}
		mem->depth_buf_bus = res.busAddr();
	} else {
		// pinned host memory for transfer chunk from GPU
		if (!mem->depth_buf.allocate(depth_buf_len, nullptr, false)) {
			return false;
		}
	}

	auto trans_buf_len = n_pixels_depth * sizeof(uint16_t);
	if (cmem) {
		host_mem_resource res = cmem->claimHostRes2(trans_buf_len);
		if (!res.map_size) {
			return false;
		}
		if (!mem->trans_buf.allocate(trans_buf_len, res.virtualAddr(), false)) {
			return false;
		}
		mem->trans_buf_bus = res.busAddr();
	} else {
		// pinned host memory for transfer chunk from GPU
		if (!mem->trans_buf.allocate(trans_buf_len, nullptr, false)) {
			return false;
		}
	}
	res_pool.push(mem);
	return true;
}

bool MeshReader::init(unsigned num_img_bufs, const string &ptx_file,
		unsigned img_width, unsigned img_height, unsigned depths_per_image,
		unsigned depth_width, unsigned depth_height)
{
	camera_img_dim.x = img_width;
	camera_img_dim.y = img_height;

	depth_maps_per_image = depths_per_image;
	camera_depth_dim.x = depth_width;
	camera_depth_dim.y = depth_height;

	int res_needed = num_img_bufs + num_img_bufs * depth_maps_per_image;
	for (int i = 0; i < res_needed; i++) {
		if (!addMem(img_width, img_height, depths_per_image, depth_width,
				depth_height)) {
			return false;
		}
	}

	ptx = loadPtxFile(ptx_file);
	if (ptx.empty()) {
		return false;
	}

	int period_ms = static_cast<int>(1000.0 / play_fps);
	timeout_dispatcher->addTimeoutMs(period_ms, this, period_ms);
	return true;
}

float2 mapPosToCamera_(float3 p, const float *m3x4)
{
	float xs = m3x4[0 * 4 + 0] * p.x + m3x4[0 * 4 + 1] * p.y
			+ m3x4[0 * 4 + 2] * p.z + m3x4[0 * 4 + 3];
	float ys = m3x4[1 * 4 + 0] * p.x + m3x4[1 * 4 + 1] * p.y
			+ m3x4[1 * 4 + 2] * p.z + m3x4[1 * 4 + 3];
	float w = m3x4[2 * 4 + 0] * p.x + m3x4[2 * 4 + 1] * p.y
			+ m3x4[2 * 4 + 2] * p.z + m3x4[2 * 4 + 3];
	return make_float2(xs / w, ys / w);
}

// like blender on export "Y up"
float3 swizzle_yup_location(float3 loc)
{
	return make_float3(loc.x, loc.z, -loc.y);
}

void test_point(float X, float Y, float Z, float *m3x4)
{
	float3 p = make_float3(X, Y, Z);
	p = swizzle_yup_location(p);
	float2 uv = mapPosToCamera_(p, m3x4);
	logc_printf(LOGC_DEBUG, "map: %fx%fx%f->%fx%f\n", p.x, p.y, p.z, uv.x,
			uv.y);
}

void dump_4x4(const string &prefix, sutil::Matrix4x4 m4x4)
{
	logc_printf(LOGC_DEBUG, "%s", prefix.c_str());
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			logc_printf(LOGC_DEBUG, "%5.3f ", m4x4[i * 4 + j]);
		}
		logc_printf(LOGC_DEBUG, "\n");
	}
}

void dump_3x4(const string &prefix, sutil::Matrix3x4 m3x4)
{
	logc_printf(LOGC_DEBUG, "%s", prefix.c_str());
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			logc_printf(LOGC_DEBUG, "%5.1f ", m3x4[i * 4 + j]);
		}
		logc_printf(LOGC_DEBUG, "\n");
	}
}

void dump_3x3(const string &prefix, sutil::Matrix3x3 m3x3)
{
	logc_printf(LOGC_DEBUG, "%s", prefix.c_str());
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			logc_printf(LOGC_DEBUG, "%5.1f ", m3x3[i * 3 + j]);
		}
		logc_printf(LOGC_DEBUG, "\n");
	}
}

void build3x4(float fx, float u_center, float v_center,
		const sutil::Matrix4x4 &trasform_matrix, float *m3x4)
{
	float K[3 * 3] = { fx, 0.0, u_center, 0.0, fx, v_center, // square pixel
					   0.0, 0.0, 1.0 };
	sutil::Matrix3x3 Km(K);
	dump_3x3("Km:\n", Km);
	// no scale for camera (scale =1.0) => this 3x3 is only rotation
	//	sutil::Matrix3x3 r = make_matrix3x3(trasform_matrix);
	sutil::Matrix4x4 rt_4x4 = trasform_matrix.inverse();
	dump_4x4("rt_4x4\n", rt_4x4);
	sutil::Matrix3x4 rt;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			rt[i * 4 + j] = rt_4x4[i * 4 + j];
			//			rt[i*4+j] = trasform_matrix[i*4+j];
		}
	}
	sutil::Matrix3x4 cm = Km * rt;
	dump_3x4("cm\n", cm);
	memcpy(m3x4, cm.getData(), sizeof(float) * 12);
}

bool MeshReader::openFile(const string &infile)
{
	if (!scene.loadScene(infile)) {
		return false;
	}
	scene.finalize(ptx);

	auto fallback_cam = make_shared<sutil::Camera>();
	fallback_cam->setFovY(45.0f);
	fallback_cam->setLookat(scene.aabb().center());
	fallback_cam->setEye(scene.aabb().center()
			+ make_float3(0.0f, 0.0f, 1.5f * scene.aabb().maxExtent()));

	camera_img_ = scene.cameraByName("Camera", fallback_cam);
	logc_printf(LOGC_DEBUG, "camera_img fov: %f\n", camera_img_->fovY());

	// we need 3x4 camera_img for fill hall_to_img_buffer
	{
		float fx = (camera_img_dim.y / 2.0)
				/ tan(deg2rad(camera_img_->fovY()) / 2.0);
		float u_center = camera_img_dim.x / 2.0;
		float v_center = camera_img_dim.y / 2.0;
		sutil::Matrix4x4 rt = camera_img_->node_xform;
		dump_4x4("camera_img.node_xform:\n", rt);
		build3x4(fx, u_center, v_center, rt, m3x4_scene_to_img);
		test_point(19.7, 0.0, 0.1, m3x4_scene_to_img);
		test_point(19.7, -0.2, 0.1, m3x4_scene_to_img);
	}

	camera_depth_ = scene.cameraByName("CameraDepth", camera_img_);
	logc_printf(LOGC_DEBUG, "camera_depth fov: %f\n", camera_depth_->fovY());
	{
		float fx = (camera_depth_dim.y / 2.0)
				/ tan(deg2rad(camera_depth_->fovY()) / 2.0);
		float u_center = camera_depth_dim.x / 2.0;
		float v_center = camera_depth_dim.y / 2.0;
		sutil::Matrix4x4 rt = camera_depth_->node_xform;
		dump_4x4("camera_depth.node_xform:\n", rt);
		float m3x4_laser[12];
		build3x4(fx, u_center, v_center, rt, m3x4_laser);
		test_point(0.3, 0.0, 0, m3x4_laser);
	}

	env_tex_ = sutil::loadFileToTex(infile + ".env");

	float loffset = scene.aabb().maxExtent();
	LOGC(DEBUG) << "aabb().maxExtent(): " << loffset << endl;

	// TODO: add light support to sutil::Scene
	vector<Light::Point> lights;
	lights.push_back(Light::Point());
	lights[0].color = { 1.0f, 1.0f, 0.8f };
	lights[0].intensity = 5.0f;
	lights[0].position = scene.aabb().center() + make_float3(loffset);
	lights[0].falloff = Light::Falloff::QUADRATIC;

//    lights.push_back(Light::Point());
//    lights[1].color     = { 0.8f, 0.8f, 1.0f };
//    lights[1].intensity = 3.0f;
//    lights[1].position  = scene.aabb().center() + make_float3( -loffset, 0.5f*loffset, -0.5f*loffset  );
//    lights[1].falloff   = Light::Falloff::QUADRATIC;

//    lights.push_back(Light::Point());
//    lights[2].color     = { 0.8f, 0.8f, 1.0f };
//    lights[2].intensity = 0.8f;
//    lights[2].position  = scene.aabb().center() + make_float3( loffset, 2.0f, -10.0f  );
//    lights[2].falloff   = Light::Falloff::QUADRATIC;

	lights_bv.count = static_cast<uint32_t>( lights.size());
	cudaCheck(cudaMalloc((void **) &lights_bv.data,
			lights.size() * sizeof(Light::Point)));
	cudaCheck(cudaMemcpy((void *) lights_bv.data, lights.data(),
			lights.size() * sizeof(Light::Point), cudaMemcpyHostToDevice));

	timeval tv;
	gettimeofday(&tv, nullptr);
	uint64_t
			t_now_us = ((uint64_t) tv.tv_sec) * 1000000 + (uint64_t) tv.tv_usec;
	time_beg_us = t_now_us;
	time_now_us = time_beg_us;
	return true;
}

float3 MeshReader::getCameraRotation() const
{
	return camera_img_->getEulerAngles();
}

void MeshReader::rotateCamEuler(float3 angles)
{
	camera_img_->rotateEulerAngles(angles);
	camera_depth_->rotateEulerAngles(angles);
}

bool MeshReader::nextFrame(bool is_needed_release)
{
	//ret false is eof
	PerfCounter::incrementCounter("mesh_nextFrame");
	if (!scene.traversableHandle()) {
		PerfCounter::incrementCounter("mesh_no_scene");
		return false;
	}
	// update time;
	time_now_us += 1000000.0 / play_fps;

	unsigned mem_needed = 1;
	mem_needed += depth_maps_per_image;
	if (res_pool.size() < mem_needed) {
		PerfCounter::incrementCounter("mesh_no_descr");
		return true;
	}
	compilation *c = nullptr;
	if (depth_maps_per_image > 0) {
		c = new struct compilation;
		c->num_awaited_depths = depth_maps_per_image;
		c->img_mem = nullptr;
		c->img_done = false;
		compilations.insert(c);
	}

	uint64_t depth_diff_us =
			(1000000.0 / play_fps) / (double) depth_maps_per_image;
	for (unsigned i = 0; i < mem_needed; i++) {
		struct sc_mem_res *mem = res_pool.front();
		res_pool.pop();

		mem->time_us = time_now_us;
		if (i > 0)
			mem->time_us = time_now_us + depth_diff_us * (i - 1);
		scene.doAnimation(mem->time_us); // change scene.m_meshes transform
		scene.rebuildInstanceAccelAsync(mem->stream.get(),
				whitted::RAY_TYPE_COUNT); // scene.m_ias_handle changed

		whitted::LaunchParams params;
		params.env_tex = 0;
		if (env_tex_)
			params.env_tex = env_tex_->get();
		params.subframe_index = 0u;
		params.miss_color = make_float3(0.1f);
		params.lights = lights_bv;

		params.plight.emission = make_float3(15.0f, 15.0f, 5.0f);
		params.plight.corner = make_float3(343.0f, 548.5f, 227.0f);
		params.plight.v1 = make_float3(0.0f, 0.0f, 105.0f);
		params.plight.v2 = make_float3(-130.0f, 10.0f, 10.0f);
		params.plight.normal =
				normalize(cross(params.plight.v1, params.plight.v2));

		if (i == 0) {
			mem->launch_width = (unsigned)camera_img_dim.x;
			mem->launch_height = (unsigned)camera_img_dim.y;
			mem->compilation = nullptr;
			if (c) {
				mem->compilation = c;
				c->img_mem = mem;
			}
			camera_img_->setAspectRatio(static_cast<float>(mem->launch_width)
					/ static_cast<float>(mem->launch_height ));
			params.eye = camera_img_->eye();
			camera_img_->UVWFrame(params.U, params.V, params.W); //gen params
			mem->camera_fov_y = camera_img_->fovY();

			params.frame_buffer =
					(unsigned char *) mem->color_buf.getDeviceBuffer();
			params.dist_buffer = nullptr;
			params.hall_to_img_buffer = nullptr;
		} else {
			mem->launch_width = (unsigned)camera_depth_dim.x;
			mem->launch_height = (unsigned)camera_depth_dim.y;
			mem->compilation = c;

			camera_depth_->setAspectRatio(
					static_cast<float>( mem->launch_width )
							/ static_cast<float>( mem->launch_height ));
			params.eye = camera_depth_->eye();
			camera_depth_->UVWFrame(params.U, params.V, params.W); //gen params
			mem->camera_fov_y = camera_depth_->fovY();

			params.frame_buffer = nullptr;
			params.dist_buffer = (uint16_t *) mem->depth_buf.getDeviceBuffer();
			params.hall_to_img_buffer =
					(uint16_t *) mem->trans_buf.getDeviceBuffer();

			// we need 3x4 fill hall_to_img_buffer
			memcpy(params.img_3x4, m3x4_scene_to_img,
					sizeof(m3x4_scene_to_img));
		}
		params.accum_buffer = nullptr;

		params.handle = scene.traversableHandle();

        mem->params_cumem.copyFromHostSync(&params, sizeof(whitted::LaunchParams));
		optixCheck(optixLaunch(scene.pipeline(), mem->stream.get(),
				reinterpret_cast<CUdeviceptr>(mem->params_cumem.get()),
				sizeof(whitted::LaunchParams), scene.sbt(),
				mem->launch_width,  // launch width
				mem->launch_height, // launch height
				1       // launch depth?
		));

		if (params.frame_buffer) {
			mem->color_buf.deviceToHost(mem->stream);
		}
		if (params.dist_buffer) {
			mem->depth_buf.deviceToHost(mem->stream);
		}
		if (params.hall_to_img_buffer) {
			mem->trans_buf.deviceToHost(mem->stream);
		}
		mem->finish_event.record(mem->stream);

		PerfCounter::incrementTotalCounter("mesh_run_q", 1);
		run_q.push(mem);
	}
	return true;
}

unsigned MeshReader::pendedFrames()
{
	return run_q.size();
}

void MeshReader::setCamDirection(float3 dirWS)
{
	camera_img_->setDirection(-dirWS);
	camera_depth_->setDirection(-dirWS);
}

void MeshReader::renderDone(struct sc_mem_res *mem_done)
{
	PerfCounter::incrementCounter("mesh_done");

	struct sc_mem_res *mem_img = nullptr;
	vector<struct sc_mem_res *> depths;
	auto ptr = compilations.find(mem_done->compilation);
	if (ptr != compilations.end()) { // use compilations
		struct compilation *c = *ptr;
		if (mem_done == c->img_mem) {
			PerfCounter::incrementCounter("mesh_done_img");
			// image render done
			c->img_done = true;
			if (c->depths.size() < c->num_awaited_depths) {
				return;
			}
			// done, send
		} else {
			PerfCounter::incrementCounter("mesh_done_depth");
			// depth render done
			c->depths.push_back(mem_done);
			if (c->depths.size() < c->num_awaited_depths) {
				return;
			}
			if (!c->img_done) {
				return;
			}
			// done, send
		}
		// compilation done
		compilations.erase(ptr);
		mem_img = c->img_mem;
		depths = c->depths;
		delete c;
	} else { // only image rendered
		mem_img = mem_done;
		PerfCounter::incrementCounter("mesh_done_img");
	}

	struct timeval capturingtime = { 0 };
	capturingtime.tv_sec = mem_img->time_us / 1000000;
	capturingtime.tv_usec = mem_img->time_us % 1000000;
	struct timespec capturingclock = { 0 };
	capturingclock.tv_sec = mem_img->time_us / 1000000;
	capturingclock.tv_nsec = (mem_img->time_us % 1000000) * 1000;

	auto gframe = new MeshReaderGrabbedFrameDescriptor(this, capturingtime,
			capturingclock, mem_img);
	gframe->frameticks =
			(uint32_t) (mem_img->time_us / 100 * 9); // 1000000 Hz -> 90000 Hz;
	gframe->pixel_width = 3.45 / 1000.0; // i like Imx265
	gframe->pixel_height = gframe->pixel_width;
	gframe->time_synced = 1;
	double zoom_mm = (gframe->pixel_height * mem_img->launch_height / 2)
			/ tan(deg2rad(mem_img->camera_fov_y) / 2.0);
	gframe->real_zoom_pos = zoom_mm;
	gframe->block_serial = "gplay";

	gframe->setDepths(depths);
	//ToDO:
	//	gframe->matrix3x3
	//	gframe->matrix3x4

	if (!depths.empty()) { // debug
		gframe->laser_meas = new lira_laser_meas;
		uint16_t *d_buf = (uint16_t *) depths[0]->depth_buf.getHostBuffer();
		unsigned laser_point_x = depths[0]->launch_width / 2;
		unsigned laser_point_y = depths[0]->launch_height / 2;
		uint16_t d_packed =
				d_buf[depths[0]->launch_width * laser_point_y + laser_point_x];
		gframe->laser_meas->range_m = ((float) (d_packed & 0x7ff)) * 0.075f;
		gframe->laser_meas->err_range_m = 0.075;
		gframe->laser_meas->speed_kmh = 0.0; //?
		gframe->laser_meas->err_speed_kmh = 100.0;
		std::ostringstream s;
		for (unsigned i = 0; i < depths.size(); i++) {
			uint16_t *d_buf = (uint16_t *) depths[i]->depth_buf.getHostBuffer();
			uint16_t d_packed = d_buf[depths[i]->launch_width * laser_point_y
					+ laser_point_x];
			s << (d_packed & 0x7ff) << ",";
		}
		printf("center range_m: %s\n", s.str().c_str());
//		printf("center range_m: %f\n", gframe->laser_meas->range_m);
		//ToDO: for emu we can do gframe->laser_rawdata here ...
	}

	distributeFrame(gframe);
	gframe->releaseObject();
}

void MeshReader::timeoutHappened(int)
{
	while (!run_q.empty()) {
		struct sc_mem_res *mem = run_q.front();
		if (!mem->finish_event.isReady()) {
			return;
		}
		run_q.pop();
		PerfCounter::incrementTotalCounter("mesh_run_q", -1);
		renderDone(mem);
	}
}

void MeshReader::reenqueMem(struct sc_mem_res *mem,
		const vector<sc_mem_res *> &depths)
{
	res_pool.push(mem);
	for (auto m: depths) {
		res_pool.push(m);
	}
}

bool MeshReader::isClaimAllowed( int thr )
{
	return (int)res_pool.size() >= thr;
}
