//
// Created by bond on 22.11.2019.
//

#include "HumanHand.h"
#include "PerfCounter.h"
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

//==============================================================================
//===================== HumanHand ================================
//==============================================================================
HumanHand::HumanHand(MeshReader *reader)
{
	this->reader = reader;
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

bool HumanHand::setCalibrate(std::string calibrate_str)
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

struct ImuSpiData fix_imu(struct ImuSpiData bad)
{
	struct ImuSpiData ok;
	ok.gyro_x = bad.gyro_y;
	ok.gyro_y = bad.gyro_x;
	ok.gyro_z = bad.accel_x;
	ok.accel_x = bad.gyro_z;
	ok.accel_y = bad.accel_z;
	ok.accel_z = bad.accel_y;
	return ok;
}

static void dumpMean(const std::deque<struct ImuSpiData> &history)
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
		w_x_sum += d.gyro_x;
		w_y_sum += d.gyro_y;
		w_z_sum += d.gyro_z;

		a_x_sum += d.accel_x;
		a_y_sum += d.accel_y;
		a_z_sum += d.accel_z;
	}
	printf("gyro_x_mean: %5d, gyro_y_mean: %5d, gyro_z_mean: %5d\n",
			w_x_sum / (int) history.size(), w_y_sum / (int) history.size(),
			w_z_sum / (int) history.size());
	printf("acc_x_mean: %5d, acc_y_mean: %5d, acc_z_mean: %5d\n",
			a_x_sum / (int) history.size(), a_y_sum / (int) history.size(),
			a_z_sum / (int) history.size());
}

void HumanHand::calcGyroCalib(const std::deque<struct ImuSpiData> &history)
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
		double w_x = (((double) d.gyro_x) / 32768.0) * 1000.0 + gyro_x_offset;
		double w_y = (((double) d.gyro_y) / 32768.0) * 1000.0 + gyro_y_offset;
		double w_z = (((double) d.gyro_z) / 32768.0) * 1000.0 + gyro_z_offset;
		double a_x = (((double) d.accel_x) / 32768.0) * 16.0 + accel_x_offset;
		double a_y = (((double) d.accel_y) / 32768.0) * 16.0 + accel_y_offset;
		double a_z = (((double) d.accel_z) / 32768.0) * 16.0 + accel_z_offset;
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

void HumanHand::calibrateTick(struct ImuSpiData d)
{
	if (!self_calibrate_needed)
		return;
	history.push_back(d);
	if (history.size() < 500) // ~ one second
		return;
	dumpMean(history);
	// do calib
	calcGyroCalib(history);
	history.clear();
	self_calibrate_needed = false;
}

static int cnt = 0;

void HumanHand::frameGrabbed(GrabbedFrameDescriptor *frame)
{
	PerfCounter::incrementCounter("imudata", frame->imuspi_data.size());
	for (unsigned i = 0; i < frame->imuspi_data.size(); i++) {
		cnt++;
		if (cnt % 3 != 0)
			continue;
		struct ImuSpiData d = frame->imuspi_data[i];
		float w_x = (((double) d.gyro_x) / 32768.0) * 1000.0 + gyro_x_offset;
		float w_y = (((double) d.gyro_y) / 32768.0) * 1000.0 + gyro_y_offset;
		float w_z = (((double) d.gyro_z) / 32768.0) * 1000.0 + gyro_z_offset;
		float a_x = (((double) d.accel_x) / 32768.0) * 16.0 + accel_x_offset;
		float a_y = (((double) d.accel_y) / 32768.0) * 16.0 + accel_y_offset;
		float a_z = (((double) d.accel_z) / 32768.0) * 16.0 + accel_z_offset;
		calibrateTick(d);

		filterUpdate(deg2rad(w_x), deg2rad(w_y), deg2rad(w_z), a_x, a_y, a_z,
				q);
	}

	sutil::Quaternion qq(make_float4(q[1], q[2], q[3], q[0]));
	float3 ea = qq.eulerAngles();

	PerfCounter::incrementAverage("hh_roll", rad2deg(ea.x));
	PerfCounter::incrementAverage("hh_pitch", rad2deg(ea.y));
	PerfCounter::incrementAverage("hh_yaw", rad2deg(ea.z));

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

