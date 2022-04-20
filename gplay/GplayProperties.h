#ifndef __GPLAY_PROPERTIES_H
#define __GPLAY_PROPERTIES_H

#include "ProgramProperties.h"

class GplayProperties: protected ProgramProperties
{
private:
	GplayProperties()
	{
		exitnow = false;
	}

	static GplayProperties *exemplar;

public:
	static GplayProperties *getInstance()
	{
		if (!exemplar)
			exemplar = new GplayProperties();
		return exemplar;
	}

	void setExitNow()
	{
		exitnow = true;
	}

	bool isExitNow()
	{
		return exitnow;
	}

	double getPlayFrameRate() // с каким framerate проигрывать
	{
		return getEnvVarAsDouble("PLAY_FRAMERATE", 25.0);
	}

	std::string getRecordFile()
	{
		return getEnvVarAsString("PLAY_ALL_FILE", "/home/dipet/scene/44.glb");
	}

	unsigned short getTCPLiveVideoListeningPort()
	{
		return getEnvVarAsInt("PLAY_LIVEVIDEOPORT", 4150);
	}

	std::string playFormat()
	{
		return getEnvVarAsString("PLAY_FORMAT", "SAVED");
	}

	std::string mailboxFrameAoi()
	{
		return getEnvVarAsString("PLAY_MAILBOXAOI", ""); // "0,700,4096x1060"
	}

	std::string mailboxVideoPlayerDevice()
	{
		return getEnvVarAsString("PLAY_MAILBOXVIDEOPLAYER",
				"/dev/emuxil_videoplay0");
	}

	bool useMailboaxAsPlayTimer()
	{
		return getEnvVarAsBool("PLAY_MAILBOXVIDEOPLAYERISTIMER", false);
	}

	int getMailboxVideoPlayerBuffs()
	{
		return getEnvVarAsInt("PLAY_MAILBOXVIDEOPLAYERBUFFS", 4);
	}

	int getMailboxVideoPlayerTargetFrame()
	{
		return getEnvVarAsInt("PLAY_MAILBOXVIDEOPLAYERTARGETFRAME", 0);
	}

	bool forceMailboxVideoPlayerGrayscale()
	{
		return getEnvVarAsBool("PLAY_MAILBOXVIDEOPLAYERFORCEGRAY", false);
	}

	bool forceMailboxVideoPlayerY0Y1()
	{
		return getEnvVarAsBool("PLAY_MAILBOXVIDEOPLAYERFORCEY0Y1", false);
	}

	int getMailboxVideoPlayerOnePice()
	{
		return getEnvVarAsInt("PLAY_MAILBOXVIDEOPLAYERONEPICE",
				8 * 1024 * 1024);
	}

	int getMailboxVideoPlayerFillThreshold()
	{
		return getEnvVarAsInt("PLAY_MAILBOXVIDEOPLAYERFILLTHRESHOLD", 5);
	}

	bool doFrameticksFromCapturingTime()
	{
		return getEnvVarAsBool("PLAY_DOFRAMETICKSFROMCAPTURINGTIME", true);
	}

	std::string ddrHeap()
	{
		return getEnvVarAsString("PLAY_DDRHEAP", "");
	}

	std::string fpgaResize0Dev()
	{
		//"/dev/arteria_resize0"
		return getEnvVarAsString("PLAY_RESIZE0DEVFPGA", "");
	}

	std::string fpgaDJpegDev()
	{
		return getEnvVarAsString("PLAY_DJPEGDEVFPGA",
				"/dev/arteria_djpeg0,/dev/arteria_djpeg1,/dev/arteria_djpeg2");
	}

	std::string fpgaDmaToFpgaDev()
	{
		return getEnvVarAsString("PLAY_DMATOFPGADEVFPGA",
				"/dev/arteria_sgdmatofpga0");
	}

	std::string fpgaPaintDev()
	{
		return getEnvVarAsString("PLAY_PAINTDEVFPGA",
				""); // /dev/arteria_paint0
	}

	int softConvThreads()
	{
		return getEnvVarAsInt("PLAY_CONVTHR", 6);
	}

	bool enableTlPainter()
	{
		return getEnvVarAsBool("PLAY_ENABLE_TL_PAINTER", false);
	}

	u_int32_t gyroAlsAddr()
	{
		return getEnvVarAsIPAddress("PLAY_GYROALSADDR", INADDR_LOOPBACK);
	}

	unsigned short gyroAlsPort()
	{
		return getEnvVarAsInt("PLAY_GYROALSPORT", 0);
	}

	std::string gyroCalibrate()
	{
		return getEnvVarAsString("PLAY_GYROCALIBRATE", "");
	}

	std::string ptxFile()
	{
		return getEnvVarAsString("PLAY_PTX_FILE",
				"/home/dipet/Projects/PhotoRadarSources/utils/gplay/cuda/1050.ptx");
	}

	bool isCopyDepthsToCmem(bool copy_depths_to_cmem_defval)
	{
		return getEnvVarAsBool("PLAY_IS_COPY_DEPTHS_TO_CMEM",
				copy_depths_to_cmem_defval);
	}

	int numRenderBuffs()
	{
		return getEnvVarAsInt("PLAY_RENDER_BUFS", 12);
	}

	int renderImageWidth()
	{
		return getEnvVarAsInt("PLAY_RENDER_IMAGE_WIDTH", 1920);
	}

	int renderImageHeight()
	{
		return getEnvVarAsInt("PLAY_RENDER_IMAGE_HEIGHT", 1080);
	}

	int numDepthsPerImage()
	{
		return getEnvVarAsInt("PLAY_DEPTHS_PER_IMAGE", 0); // 20
	}

	int renderDepthWidth()
	{
		return getEnvVarAsInt("PLAY_RENDER_DEPTH_WIDTH", 512);
	}

	int renderDepthHeight()
	{
		return getEnvVarAsInt("PLAY_RENDER_DEPTH_HEIGHT", 512);
	}

	unsigned short getControlListeningPort()
	{
		return getEnvVarAsInt("PLAY_CONTROL_PORT", 0);
	}

	int webDisplayPort()
	{
		return getEnvVarAsInt("PLAY_WEBDISPLAY_PORT", 0);
	}

	std::string historyFile(std::string default_file)
	{
		return getEnvVarAsString("PLAY_HISTORYFILE", default_file);
	}

	std::string recordsDir()
	{
		return getEnvVarAsString("PLAY_RECORDS_DIR", ".");
	}

	std::string imuDev()
	{
		return getEnvVarAsString("PLAY_IMU_DEV", "");
	}


protected:
	bool exitnow;
};

#endif
