//
// Created by bond on 22.11.2019.
//

#ifndef PHOTORADARSOURCES_GYROALSCONNECTINGSOCK_H
#define PHOTORADARSOURCES_GYROALSCONNECTINGSOCK_H

#include "VideoFrameGrabber.h"
#include "MeshReader.h"
#include <deque>

class HumanHand: public GrabberFrameListener
{
public:
	HumanHand(MeshReader *reader);

	virtual ~HumanHand()
	{
	}

	// own interce
	bool setCalibrate(std::string calibrate_str);

	// GrabberFrameListener interface
	void frameGrabbed(GrabbedFrameDescriptor *frame);
private:
	void calibrateTick(struct ImuSpiData d);
	void calcGyroCalib(const std::deque<struct ImuSpiData> &history);
private:
	MeshReader *reader;

	float q[4];

	double gyro_x_offset;
	double gyro_y_offset;
	double gyro_z_offset;
	double accel_x_offset;
	double accel_y_offset;
	double accel_z_offset;

	std::deque<struct ImuSpiData> history;
	bool self_calibrate_needed;
};


#endif //PHOTORADARSOURCES_GYROALSCONNECTINGSOCK_H
