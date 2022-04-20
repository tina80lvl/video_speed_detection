/*
 * HumanHandImu.h
 *
 *  Created on: 10.05.2020
 *      Author: bond
 */

#ifndef GPLAY_HUMANHANDIMU_H_
#define GPLAY_HUMANHANDIMU_H_

#include "TimeoutDispatcherFd.h"
#include "MeshReader.h"

struct sOneValue
{
	uint32_t time_90k; // значение счетчика 90КГц
	uint32_t value_y;  // Задержка отклика лазера в тактах АЦП (0.5 нс)
	uint32_t reserved; // зарезервировано
	uint32_t value_x;  // Порядковый номер данных

	// значения датчиков холла
	int16_t hall_data_y;
	int16_t hall_data_x;

	// Значения гироскопов и акселерометров
	int16_t gyro_data_x;
	int16_t acc_data_x;

	int16_t gyro_data_y;
	int16_t acc_data_y;

	int16_t gyro_data_z;
	int16_t acc_data_z;
}__attribute__((packed));

class HumanHandImu: public TimeoutHandler
{
public:
	HumanHandImu(TimeoutDispatcherFd *timeoutdispatcher, MeshReader *reader);
	~HumanHandImu() override;

	//own interface
	bool init(const std::string &dev);
	bool setCalibrate(const std::string &calibrate_str);
	void startSelfCalib();
	void enable();
	void disable();


	//TimeoutHandler interface
	void timeoutHappened(int timeout_id) override;

private:
	void calibrateTick(struct sOneValue d);
	void calcGyroCalib();

private:
	TimeoutDispatcherFd *timeoutdispatcher;
	MeshReader *reader;

	int timer_id;

	int imuspi_fd;
	uint32_t adc_round_buf_len;
	void *adc_buf;
	int prev_idx;

	int sample_cnt;
	float q[4];
	double gyro_x_offset;
	double gyro_y_offset;
	double gyro_z_offset;
	double accel_x_offset;
	double accel_y_offset;
	double accel_z_offset;

	bool self_calibrate_needed;
	std::deque<struct sOneValue> history;
};

#endif /* GPLAY_HUMANHANDIMU_H_ */
