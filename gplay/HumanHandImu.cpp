/*
 * ImuSpiReceiver.cpp
 *
*  Created on: 10.05.2020
 *      Author: bond
 */

#include "HumanHandImu.h"

#include "PerfCounter.h"
#include "Log.h"
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#include "sutil/Quaternion.h"
#include "math_utils.h"

// https://x-io.co.uk/res/doc/madgwick_internal_report.pdf

// Math library required for ‘sqrt’
#include <math.h>

// System constants
// 833 Hz -> samplin rate of IMU, 1/833 ~ 1.2 ms
// 500Hz -> fpga read speed, 1/500 = 2 ms
// 1.2*5 = 2*3
// get each %3 value readed by fpga, it is %5 value readed by IMU

#define deltat (1.0/833.0)*5.0

#define gyroMeasError M_PI * (-1.0f / 180.0f)
#define beta 0.000033 // 0.033/1000 wtf?!

static void
filterUpdate(float w_x, float w_y, float w_z, float a_x, float a_y, float a_z,
		float *q)
{
	float SEq_1 = q[0];
	float SEq_2 = q[1];
	float SEq_3 = q[2];
	float SEq_4 = q[3];

// Local system variables
	float norm; // vector norm
	float SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3,
			SEqDot_omega_4; // quaternion derrivative from gyroscopes elements
	float f_1, f_2, f_3; // objective function elements
	float J_11or24, J_12or23, J_13or22, J_14or21, J_32,
			J_33; // objective function Jacobian elements
	float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3,
			SEqHatDot_4; // estimated direction of the gyroscope error
// Axulirary variables to avoid reapeated calcualtions
	float halfSEq_1 = 0.5f * SEq_1;
	float halfSEq_2 = 0.5f * SEq_2;
	float halfSEq_3 = 0.5f * SEq_3;
	float halfSEq_4 = 0.5f * SEq_4;
	float twoSEq_1 = 2.0f * SEq_1;
	float twoSEq_2 = 2.0f * SEq_2;
	float twoSEq_3 = 2.0f * SEq_3;

// Normalise the accelerometer measurement
	norm = sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
	if (norm == 0.0)
		return;
	a_x /= norm;
	a_y /= norm;
	a_z /= norm;
// Compute the objective function and Jacobian
	f_1 = twoSEq_2 * SEq_4 - twoSEq_1 * SEq_3 - a_x;
	f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4 - a_y;
	f_3 = 1.0f - twoSEq_2 * SEq_2 - twoSEq_3 * SEq_3 - a_z;
	J_11or24 = twoSEq_3; // J_11 negated in matrix multiplication
	J_12or23 = 2.0f * SEq_4;
	J_13or22 = twoSEq_1; // J_12 negated in matrix multiplication
	J_14or21 = twoSEq_2;
	J_32 = 2.0f * J_14or21; // negated in matrix multiplication
	J_33 = 2.0f * J_11or24; // negated in matrix multiplication
// Compute the gradient (matrix multiplication)
	SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1;
	SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3;
	SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1;
	SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;
// Normalise the gradient
	norm = sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2
			+ SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
	if (norm == 0.0)
		return;
	SEqHatDot_1 /= norm;
	SEqHatDot_2 /= norm;
	SEqHatDot_3 /= norm;
	SEqHatDot_4 /= norm;
// Compute the quaternion derrivative measured by gyroscopes
	SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z;
	SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y;
	SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x;
	SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x;
// Compute then integrate the estimated quaternion derrivative
	SEq_1 += (SEqDot_omega_1 - (beta * SEqHatDot_1)) * deltat;
	SEq_2 += (SEqDot_omega_2 - (beta * SEqHatDot_2)) * deltat;
	SEq_3 += (SEqDot_omega_3 - (beta * SEqHatDot_3)) * deltat;
	SEq_4 += (SEqDot_omega_4 - (beta * SEqHatDot_4)) * deltat;
// Normalise quaternion
	norm = sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
	if (norm == 0.0)
		return;
	SEq_1 /= norm;
	SEq_2 /= norm;
	SEq_3 /= norm;
	SEq_4 /= norm;

	q[0] = SEq_1;
	q[1] = SEq_2;
	q[2] = SEq_3;
	q[3] = SEq_4;
}

HumanHandImu::HumanHandImu(TimeoutDispatcherFd *timeoutdispatcher,
		MeshReader *reader)
{
	this->timeoutdispatcher = timeoutdispatcher;
	this->reader = reader;
	imuspi_fd = -1;

	adc_round_buf_len = 0;
	adc_buf = (void *) -1;
	prev_idx = -1;

	sample_cnt = 0;
	q[0] = 1.0;
	q[1] = 0.0;
	q[2] = 0.0;
	q[3] = 0.0;
	gyro_x_offset = 0.0;
	gyro_y_offset = 0.0;
	gyro_z_offset = 0.0;

	accel_x_offset = 0.0;
	accel_y_offset = 0.0;
	accel_z_offset = 0.0;
	self_calibrate_needed = true;
}

HumanHandImu::~HumanHandImu()
{
	if (adc_buf != (void *) -1)
		munmap(adc_buf, adc_round_buf_len);
	if (imuspi_fd >= 0)
		close(imuspi_fd);
	if (timer_id >= 0)
		timeoutdispatcher->cancelTimeout(timer_id);
}


#define IMU_IOC_MAGIC 0x5e
#define IMU_GETADCBUFLEN _IOC(_IOC_READ, IMU_IOC_MAGIC, 1, 4)
#define IMU_STARTADCWRITE _IOC(_IOC_NONE, IMU_IOC_MAGIC, 2, 0)
#define IMU_GETLASTOFFSET _IOC(_IOC_READ, IMU_IOC_MAGIC, 9, 4)

bool HumanHandImu::init(const std::string &dev)
{
	imuspi_fd = open(dev.c_str(), O_RDWR);
	if (imuspi_fd < 0) {
		logc_printf(LOGC_ERROR, "HumanHandImu: cant open %s: %s\n", dev.c_str(),
				strerror(errno));
		return false;
	}
	int ret = ioctl(imuspi_fd, IMU_GETADCBUFLEN, &adc_round_buf_len);
	if (ret != 0) {
		logc_printf(LOGC_ERROR,
				"HumanHandImu: can't do IMUSPI_GETADCBUFLEN: %s\n",
				strerror(errno));
		return false;
	}
	logc_printf(LOGC_INFO, "HumanHandImu: buf size %u\n", adc_round_buf_len);

	adc_buf = mmap(0, adc_round_buf_len, PROT_READ, MAP_SHARED, imuspi_fd, 0);
	if (adc_buf == (void *) -1) {
		logc_printf(LOGC_ERROR,
				"HumanHandImu: cant mmap %d bytes for adc data: %s\n",
				adc_round_buf_len, strerror(errno));
		return false;
	}

	ret = ioctl(imuspi_fd, IMU_STARTADCWRITE);
	if (ret != 0) {
		logc_printf(LOGC_ERROR,
				"HumanHandImu: can't do IMUSPI_STARTADCWRITE: %s\n",
				strerror(errno));
		return false;
	}
	timer_id = -1;
	return true;
}

void HumanHandImu::enable()
{
	if (timer_id >= 0)
		return;
	logc_printf(LOGC_INFO, "HumanHandImu: enable\n");
	timer_id = timeoutdispatcher->addTimeoutMs(100, this, 100);
	prev_idx = -1;
}

void HumanHandImu::disable()
{
	if (timer_id < 0)
		return;
	logc_printf(LOGC_INFO, "HumanHandImu: disable\n");
	timeoutdispatcher->cancelTimeout(timer_id);
	timer_id = -1;
}

void HumanHandImu::startSelfCalib()
{
	if (self_calibrate_needed)
		return;
	logc_printf(LOGC_INFO, "HumanHandImu: startSelfCalib \n");
	history.clear();
	gyro_x_offset = 0.0;
	gyro_y_offset = 0.0;
	gyro_z_offset = 0.0;

	accel_x_offset = 0.0;
	accel_y_offset = 0.0;
	accel_z_offset = 0.0;
	self_calibrate_needed = true;
}

bool HumanHandImu::setCalibrate(const std::string &calibrate_str)
{
	double v[6];
	int ret = sscanf(calibrate_str.c_str(), "%lf %lf %lf %lf %lf %lf", &v[0],
			&v[1], &v[2], &v[3], &v[4], &v[5]);
	if (ret != 6)
		return false;
	gyro_x_offset = v[0];
	gyro_y_offset = v[1];
	gyro_z_offset = v[2];
	accel_x_offset = v[3];
	accel_y_offset = v[4];
	accel_z_offset = v[5];
	self_calibrate_needed = false;
	return true;
}

static void dumpMean(const std::deque<struct sOneValue> &history)
{
	if (history.empty())
		return;
	int32_t w_x_sum = 0;
	int32_t w_y_sum = 0;
	int32_t w_z_sum = 0;

	int32_t a_x_sum = 0;
	int32_t a_y_sum = 0;
	int32_t a_z_sum = 0;

	for (auto d: history) {
		w_x_sum += d.gyro_data_x;
		w_y_sum += d.gyro_data_y;
		w_z_sum += d.gyro_data_z;

		a_x_sum += d.acc_data_x;
		a_y_sum += d.acc_data_y;
		a_z_sum += d.acc_data_z;
	}
	printf("gyro_x_mean: %5d, gyro_y_mean: %5d, gyro_z_mean: %5d\n",
			w_x_sum / (int) history.size(), w_y_sum / (int) history.size(),
			w_z_sum / (int) history.size());
	printf("acc_x_mean: %5d, acc_y_mean: %5d, acc_z_mean: %5d\n",
			a_x_sum / (int) history.size(), a_y_sum / (int) history.size(),
			a_z_sum / (int) history.size());
}

void HumanHandImu::calcGyroCalib()
{
	if (history.empty())
		return;
	double w_x_sum = 0;
	double w_y_sum = 0;
	double w_z_sum = 0;

	double a_x_sum = 0;
	double a_y_sum = 0;
	double a_z_sum = 0;

	for (auto d: history) {
		double w_x =
				(((double) d.gyro_data_x) / 32768.0) * 1000.0 + gyro_x_offset;
		double w_y =
				(((double) d.gyro_data_y) / 32768.0) * 1000.0 + gyro_y_offset;
		double w_z =
				(((double) d.gyro_data_z) / 32768.0) * 1000.0 + gyro_z_offset;
		double a_x =
				(((double) d.acc_data_x) / 32768.0) * 16.0 + accel_x_offset;
		double a_y =
				(((double) d.acc_data_y) / 32768.0) * 16.0 + accel_y_offset;
		double a_z =
				(((double) d.acc_data_z) / 32768.0) * 16.0 + accel_z_offset;
		w_x_sum += w_x;
		w_y_sum += w_y;
		w_z_sum += w_z;

		a_x_sum += a_x;
		a_y_sum += a_y;
		a_z_sum += a_z;
	}
	gyro_x_offset = -w_x_sum / (double) history.size();
	gyro_y_offset = -w_y_sum / (double) history.size();
	gyro_z_offset = -w_z_sum / (double) history.size();
	q[0] = 1.0;
	q[1] = 0.0;
	q[2] = 0.0;
	q[3] = 0.0;
	printf("gyro_x_offset: %.3lf gyro_y_offset: %.3lf gyro_z_offset: %.3lf\n",
			gyro_x_offset, gyro_y_offset, gyro_z_offset);

	accel_x_offset = -a_x_sum / (double) history.size();
	/* > 0.5 работае только если калибровка в режиме когда одна ось точно больше других*/
	if (fabs(accel_x_offset) > 0.5)
		accel_x_offset -= 1.0;
	accel_y_offset = -a_y_sum / (double) history.size();
	if (fabs(accel_y_offset) > 0.5)
		accel_y_offset -= 1.0;
	accel_z_offset = -a_z_sum / (double) history.size();
	if (fabs(accel_z_offset) > 0.5)
		accel_z_offset -= 1.0;

	printf("accel_x_offset: %.3lf accel_y_offset: %.3lf accel_z_offset: %.3lf\n",
			accel_x_offset, accel_y_offset, accel_z_offset);
}

void HumanHandImu::calibrateTick(struct sOneValue d)
{
	if (!self_calibrate_needed)
		return;
	history.push_back(d);
	if (history.size() < 500) // ~ one second
		return;
	dumpMean(history);
	// do calib
	calcGyroCalib(); // fill gyro_xyz_offset, accel_xyz_offset
	history.clear();
	self_calibrate_needed = false;
}

void HumanHandImu::timeoutHappened(int)
{
	uint32_t last_offset;
	int ret = ioctl(imuspi_fd, IMU_GETLASTOFFSET, &last_offset);
	if (ret != 0) {
		logc_printf(LOGC_ERROR,
				"HumanHandImu: can't do IMUSPI_GETLASTOFFSET: %s\n",
				strerror(errno));
		exit(1);
	}
	uint32_t last_writen_idx = last_offset / sizeof(sOneValue);
	if (prev_idx < 0)
		prev_idx = last_writen_idx;
	volatile struct sOneValue *base = (volatile struct sOneValue *) adc_buf;
	uint32_t n = adc_round_buf_len / sizeof(base[0]);
	std::vector<sOneValue> imuspi_data;
	imuspi_data.reserve(n);
	uint32_t mask = (adc_round_buf_len / sizeof(base[0])) - 1;
	uint32_t beg_idx = (uint32_t)(prev_idx + 1) & mask;
	uint32_t end_idx = (last_writen_idx + 1) & mask;
	for (uint32_t i = 0; i < n; i++) {
		uint32_t idx = (beg_idx + i) & mask;
		if (idx == end_idx)
			break;
		struct sOneValue v;
		memcpy(&v, (const void *) &base[idx], sizeof(v));
		imuspi_data.push_back(v);
		prev_idx = idx;
	}
	PerfCounter::incrementCounter("imudata", imuspi_data.size());
	if (imuspi_data.empty())
		return;

	// process data
	for (unsigned i = 0; i < imuspi_data.size(); i++) {
		sample_cnt++;
		if (sample_cnt % 3 != 0)
			continue;
		struct sOneValue d = imuspi_data[i];
		float w_x = (((double) d.gyro_data_x) / 32768.0) * 1000.0
				+ gyro_x_offset;
		float w_y = (((double) d.gyro_data_y) / 32768.0) * 1000.0
				+ gyro_y_offset;
		float w_z = (((double) d.gyro_data_z) / 32768.0) * 1000.0
				+ gyro_z_offset;
		float a_x =
				(((double) d.acc_data_x) / 32768.0) * 16.0 + accel_x_offset;
		float a_y =
				(((double) d.acc_data_y) / 32768.0) * 16.0 + accel_y_offset;
		float a_z =
				(((double) d.acc_data_z) / 32768.0) * 16.0 + accel_z_offset;
		calibrateTick(d);

		filterUpdate(deg2rad(w_x), deg2rad(w_y), deg2rad(w_z), a_x, a_y,
				a_z, q);
	}

	sutil::Quaternion qq(make_float4(q[1], q[2], q[3], q[0]));
	float3 ea = qq.eulerAngles();

	PerfCounter::incrementAverage("hh_roll", (int)rad2deg(ea.x));
	PerfCounter::incrementAverage("hh_pitch", (int)rad2deg(ea.y));
	PerfCounter::incrementAverage("hh_yaw", (int)rad2deg(ea.z));

	//https://www.gamedev.net/forums/topic/56471-extracting-direction-vectors-from-quaternion/
	float3 cam_dir;

	// lira
	float3 cam_dir_ = qq.rotate(make_float3(0.0, 0.0, 1.0));
	cam_dir.x = cam_dir_.z;
	cam_dir.y = -cam_dir_.x;
	cam_dir.z = -cam_dir_.y;

	if (reader)
		reader->setCamDirection(cam_dir);
}
