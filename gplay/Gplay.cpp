#include "HumanHand.h"
#include "utils/gplay/cuda/cuda_utils.h"
#include "MeshReader.h"
#include "GplayProperties.h"

#include "FileDescriptorPoller.h"
#include "DeferredExecutor.h"
#include "PerfCounter.h"
#include "TimeoutDispatcherFd.h"
#include "VideoFrameQueue.h"
#include "PureLiveVideoListeningSocket.h"
#include "ImageFormatConverterThreaded.h"
#include "FrameTimer.h"
#include "AnalyzerConnectingSocket.h"

#include "MemoryHeap.h"
#include "MailboxVideoPlayer.h"
#include "TlPainter.h"
#include "ArteriaFrameFormatConverter.h"
#include "CMem.h"
#include "GplayWebDisplay.h"
#include "GplayControlSocket.h"
#include "HumanHandImu.h"

#include <csignal>


static CMem *cmem_instance;

CMem *getCmem()
{
	if (cmem_instance)
		return cmem_instance;
	cmem_instance = new CMem();
	cmem_instance->start();
	return cmem_instance;
}

volatile sig_atomic_t sigint_flag = 0;

void my_function(int sig) // can be called asynchronously
{
	sigint_flag = 1; // set flag
}

int main(int argc_, char *argv_[])
{
	log_init_logger2("gplay");
	GplayProperties *props = GplayProperties::getInstance();

	FileDescriptorPoller poller;
	TimeoutDispatcherFd timeoutdispatcher(&poller);
	DeferredExecutor deferredexecutor;
	PerfCounter perfcounter(&timeoutdispatcher, true);

	// читает данные из файла записи в очередь кадров
	int videoframequeuesize = 1;
	VideoFrameQueue *framequeue = new VideoFrameQueue(videoframequeuesize);

	MeshReader *reader = nullptr;
	if (props->renderImageWidth() > 0) {
		CMem *cmem = nullptr;
//		bool copy_depths_to_cmem =
//				props->isCopyDepthsToCmem(props->numDepthsPerImage() > 0);
//		if (copy_depths_to_cmem) {
//			cmem = getCmem();
//			if (!cmem) {
//				LOGC(ERROR)
//					<< "ERROR! missconfiguration, cmem needed for copy_depths_to_cmem"
//							<< std::endl;
//				exit(1);
//			}
//		}
		double play_fps = props->getPlayFrameRate();
		reader = new MeshReader(&timeoutdispatcher, play_fps, cmem);
		unsigned num_img_bufs = props->numRenderBuffs();
		bool ok = reader->init(num_img_bufs, props->ptxFile(),
				props->renderImageWidth(), props->renderImageHeight(),
				props->numDepthsPerImage(), props->renderDepthWidth(),
				props->renderDepthHeight());
		if (!ok) {
			exit(1);
		}
		reader->addFrameListener(framequeue);
	}

	ImageFormatConverterThreaded *imageformaconverterthreaded =
			new ImageFormatConverterThreaded(&poller);
	imageformaconverterthreaded->start(props->softConvThreads());

	// send video by lvds to arteria
	MemoryHeap *ddr_heap = nullptr;
	MailboxVideoPlayer *mailboxvideoplayer = nullptr;
	ArteriaFrameFormatConverter *fpgaconverter = nullptr;
	TlEmu *tl_emu = nullptr;
//	if (!props->ddrHeap().empty()) {
//		CMem *cmem = getCmem();
//		if (!cmem) {
//			LOGC(ERROR)
//				<< "ERROR! missconfiguration, cmem needed for MailboxVideoPlayer !"
//						<< std::endl;
//			exit(1);
//		}
//		ddr_heap = new MemoryHeap();
//		bool ok = ddr_heap->init(props->ddrHeap());
//		if (!ok) {
//			LOGC(ERROR)
//				<< "ERROR! Can't init ddr heap for ArteriaFrameFormatConverter!"
//						<< std::endl;
//			exit(1);
//		}
//		fpgaconverter =
//				new ArteriaFrameFormatConverter(&poller, &deferredexecutor,
//						imageformaconverterthreaded, cmem, ddr_heap, 0);
//		fpgaconverter->start("", props->fpgaResize0Dev(), "",
//				props->fpgaDJpegDev(), props->fpgaDmaToFpgaDev(), "",
//				props->fpgaPaintDev(), 0, 0, 0, "");
//
//		if (props->enableTlPainter()) {
//			tl_emu = new TlEmu(&timeoutdispatcher);
//		}
//		int targerframe = props->getMailboxVideoPlayerTargetFrame();
//		bool force_grayscale = props->forceMailboxVideoPlayerGrayscale();
//		bool force_y0y1 = props->forceMailboxVideoPlayerY0Y1();
//		int fill_thresh = props->getMailboxVideoPlayerFillThreshold();
//		bool do_frameticks_from_capturingtime =
//				props->doFrameticksFromCapturingTime();
//		mailboxvideoplayer = new MailboxVideoPlayer(&poller, &timeoutdispatcher,
//				&deferredexecutor, framequeue, fpgaconverter,
//				props->useMailboaxAsPlayTimer() ? reader : nullptr, cmem,
//				tl_emu, targerframe, force_grayscale, force_y0y1, fill_thresh,
//				do_frameticks_from_capturingtime);
//		unsigned num_buffs = props->getMailboxVideoPlayerBuffs();
//		unsigned one_pice = props->getMailboxVideoPlayerOnePice();
//		mailboxvideoplayer->start(props->mailboxVideoPlayerDevice(), ddr_heap,
//				num_buffs, one_pice);
//		if (!props->mailboxFrameAoi().empty()) {
//			mailboxvideoplayer->setAoi(
//					Rectangle::fromStr(props->mailboxFrameAoi()));
//		}
//	}

	// знает когда надо посылать след кадр
	FrameTimer *frametimer;
	FrameTimerTimeout *frametimer_timer = nullptr;
//	if (mailboxvideoplayer && props->useMailboaxAsPlayTimer()) {
//		frametimer = mailboxvideoplayer;
//		LOGC(INFO) << "use MailboxVideoPlayer as timer" << std::endl;
//	} else {
		frametimer_timer = new FrameTimerTimeout(&timeoutdispatcher, reader,
				static_cast<int>(1000.0 / props->getPlayFrameRate()));
		frametimer = frametimer_timer;
		LOGC(INFO) << "use FrameTimerTimeout as timer" << std::endl;
//	}

	// сокет показывающий видео
	PureLiveVideoListeningSocket *livevideosocket = nullptr;
	if (props->getTCPLiveVideoListeningPort() != 0) {
		livevideosocket = new PureLiveVideoListeningSocket(&deferredexecutor,
				imageformaconverterthreaded, framequeue, reader);
		livevideosocket->addToPoller(&poller);
		livevideosocket->startAcceptingConnections(
				props->getTCPLiveVideoListeningPort(), INADDR_ANY);
	}
	AnalyzerConnectingSocket *als_sock = nullptr;
	HumanHand *camera_control_als_imu = nullptr;
	if (props->gyroAlsPort() != 0) {
		als_sock = new AnalyzerConnectingSocket(&poller, &timeoutdispatcher,
				&deferredexecutor, nullptr, props->gyroAlsAddr(),
				props->gyroAlsPort(), "gyro_als", "", 50);
		als_sock->initiateConnecting();

		camera_control_als_imu = new HumanHand(reader);
		camera_control_als_imu->setCalibrate(props->gyroCalibrate());
		als_sock->addFrameListener(camera_control_als_imu);
	}
	HumanHandImu *camera_control_local_imu = 0;
	if (props->imuDev() != "") {
		camera_control_local_imu = new HumanHandImu(&timeoutdispatcher, reader);
		bool ok = camera_control_local_imu->init(props->imuDev());
		if (!ok) {
			exit(1);
		}
		camera_control_local_imu->setCalibrate(props->gyroCalibrate());
		camera_control_local_imu->enable();
	}

#ifdef HAVE_WITTY
	// отображение в webe...
	ImageConversionQueue *conv_q = 0;
	GplayWebDisplay *webdisplay = 0;
	if (props->webDisplayPort()) {
		ParamsStr convertParams;
		convertParams.setParam("DESTFORMAT", "JPEG");
		convertParams.setParam("DW", "720");
		conv_q = new ImageConversionQueue(imageformaconverterthreaded,
				convertParams);
		framequeue->addOutgoingFrameHandler(conv_q);
		webdisplay = new GplayWebDisplay(props->webDisplayPort(),
				props->getControlListeningPort(), props->recordsDir(), conv_q);
		conv_q->addFrameListener(webdisplay);
	}
#endif
	// управляет воспроизведением
	GplayControlListeningSocket *controlsock = nullptr;
	if (props->getControlListeningPort() != 0) {
		std::string historyfile = props->historyFile( props->recordsDir()  + "/.gplayhistory");
		controlsock = new GplayControlListeningSocket(&poller, &deferredexecutor,
				&timeoutdispatcher, frametimer, reader, tl_emu,
				camera_control_local_imu, historyfile);
		controlsock->addToPoller(&poller);
		controlsock->startAcceptingConnections(props->getControlListeningPort(),
				INADDR_ANY);
	}

	if (controlsock){
		bool ok = controlsock->openLastFromHistory();
		if (!ok)
			LOGC(WARNING) << "no rc for play in history, wait filename "
							 "from sock/web "<< std::endl;
		else
			frametimer->play();
	}
	else{
		std::string rc_name = props->getRecordFile();
		bool ok = reader->openFile(rc_name);
		if (!ok) {
			LOGC(ERROR) << "can't open PLAY_ALL_FILE, and no control sock, "
						   "exit" << std::endl;
			exit(1);
		}
//		if (gpslaymailbox)
//			gpslaymailbox->openFile(rc_name + ".nmea");
		frametimer->play();
	}

	signal(SIGINT, my_function);
	while (!props->isExitNow()) {
		poller.poll(1000);
		poller.execute();
		deferredexecutor.executeCommands();
		if (sigint_flag)
			break;
	}
	LOGC(INFO) << "do exit." << std::endl;
	if (reader) {
		reader->printBusyFrames();
	}
#ifdef HAVE_WITTY
	delete webdisplay;
#endif
	delete framequeue;
	delete livevideosocket;
	delete imageformaconverterthreaded;
	delete frametimer_timer;
	delete reader;
	delete GplayProperties::getInstance();
}
