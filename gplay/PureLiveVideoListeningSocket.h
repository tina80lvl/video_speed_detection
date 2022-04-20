#ifndef __PURE_LIVE_VIDEO_LISTENING_SOCKET_H
#define __PURE_LIVE_VIDEO_LISTENING_SOCKET_H

#include "ImageFormatConverter.h"
#include "FrameSequenceListener.h"
#include "VideoFrameQueue.h"
#include "TCPListeningSocket.h"
#include "TextTCPSocket.h"
#include "FrameDescriptor.h"
#include "FileDescriptorPoller.h"
#include <queue>

#include "MeshReader.h"

//=========================================================================== 
//========================= LiveVideoSocket =================================
//===========================================================================

class PureLiveVideoListeningSocket;
class PureLiveVideoSocket: public TextTCPSocket
{
public:
	PureLiveVideoSocket(int sockfd, PureLiveVideoListeningSocket *father,
			FileDescriptorPoller *poller, DeferredExecutor *deferredexecutor,
			ImageFormatConverter *imageformatconverter,
			VideoFrameQueue *framequeue, MeshReader *reader);

	~PureLiveVideoSocket();

	void stringReceived(const std::string &received);

	// Реализация интерфейса Pollable
	void disconnectHappened(int fd);
	void errorHappened(int fd);

	// Уведомление о том, что другая сторона закрыла свою сторону
	// соединения на запись
	void endOfFileEncounteredOnInput();

	// Здесь мы только проверяем, не пытаются ли нам вкинуть строку
	// мегабайтных размеров - этим можно устроить нам DoS, а мы этого не
	// хотим. Если такое случилось - ругаемся в логи и обрываем
	// соединение.
	// Для примера примем, что строка не должна быть длиннее 10К.
	void somethingWasRead();

	// Мы не читаем двоичных блоков, поэтому и на уведомление об их
	// приеме можем не реагировать
	void binaryBlockReceived(unsigned char *block, size_t size)
	{
	}

	// Нам не нужно на лету доформировывать отправляемый поток
	void nothingToSend()
	{
	}

private:
	bool setCamera(const std::vector<std::string> &received);
	void httpGet(const std::vector<std::string> &received);
	void sendCamera();

	MeshReader *mesh_reader_;

private:
	//--------------------------------------------------------------------------
	// ждет следуещий кадр и отдает его в сокет
	class SnapshotWaiter:
			public ImageFormatConverter::ResultListener,
			public FrameQueueListener
	{
	public:
		SnapshotWaiter(PureLiveVideoSocket *return_to,
				VideoFrameQueue *framequeue,
				ImageFormatConverter *imageformatconverter);
		virtual ~SnapshotWaiter();
		void request(int jpegquality, std::string snapshotsizename);
		// Очередь сдвинулась
		void frameQueueShifted(VideoFrameQueue *framequeue);
		// ImageFormatConverter interface
		void conversionFinished(int taskid, FrameDescriptor *result);

		inline void setHttpMode()
		{ http_ = true; }

		inline bool isHttp()
		{ return http_; }

	private:
		void sendFrame(GrabbedFrameDescriptor *frame);
	private:
		PureLiveVideoSocket *return_to;
		ImageFormatConverter *imageformatconverter;
		VideoFrameQueue *framequeue;

		// Качество сжатия
		int jpegquality;
		//размер изображения "","MIDDLE","SMALL"
		std::string snapshotsizename;

		std::pair<std::string, int> report;

		bool current_frame_arleady_sended;
		bool frame_on_shift_needed;
		int targerframe;
		int compressid;

		bool http_ = false;
	};

	class MjpegSender:
			public ImageFormatConverter::ResultListener,
			public FrameQueueListener
	{
	public:
		MjpegSender(PureLiveVideoSocket *return_to, VideoFrameQueue *framequeue,
				ImageFormatConverter *imageformatconverter,
				ParamsStr *compressparams);
		~MjpegSender() override;
		void begin();
		// Реализация интерфейса FrameListener для получения кадров из
		// Очередь сдвинулась
		void frameQueueShifted(VideoFrameQueue *framequeue) override;
		// ImageFormatConverter interface
		void conversionFinished(int taskid, FrameDescriptor *result) override;

	private:
		PureLiveVideoSocket *return_to;
		ImageFormatConverter *imageformatconverter;
		VideoFrameQueue *framequeue;
		ParamsStr compressparams;
		int target_frame = 0;

		std::set<int> compression_ids;
	};

protected:
	PureLiveVideoListeningSocket *father;
	ImageFormatConverter *imageformatconverter;
	VideoFrameQueue *framequeue;

	SnapshotWaiter *snapshotSendder;
	bool wasAuthenticated;

	MjpegSender *mjpeg_ = nullptr;
};

//=========================================================================== 
//========================= PureLiveVideoListeningSocket ========================
//===========================================================================
class PureLiveVideoListeningSocket: public TCPListeningSocket
{
public:
	PureLiveVideoListeningSocket(DeferredExecutor *deferredexecutor,
			ImageFormatConverter *imageformatconverter,
			VideoFrameQueue *framequeue, MeshReader *reader);
	~PureLiveVideoListeningSocket();

	// Здесь мы создаем PureLiveVideoSocket-ы
	void connectionAccepted(int acceptedsockfd, unsigned short int portfrom,
			u_int32_t ipfrom);

private:
	DeferredExecutor *deferredexecutor;
	ImageFormatConverter *imageformatconverter;
	VideoFrameQueue *framequeue;
	std::set<PureLiveVideoSocket *> connections;
	MeshReader *mesh_reader_;

	friend class PureLiveVideoSocket;

	// Сообщение от PureLiveVideoSocket-а, что его можно закрывать
	void connectionClosed(PureLiveVideoSocket *closingsocket)
	{
		connections.erase(closingsocket);
	}

};

#endif
