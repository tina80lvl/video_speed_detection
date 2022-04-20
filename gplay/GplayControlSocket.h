/*
 * GplayControlSocket.h
 *
 *      Author: bond
 */

#ifndef GPLAYCONTROLSOCKET_H_
#define GPLAYCONTROLSOCKET_H_

#include "FrameTimer.h"
#include "MeshReader.h"
#include "TextTCPSocket.h"
#include "TCPListeningSocket.h"
#include "BlockForTransfer.h"
#include "TimeoutDispatcher.h"
#include "TlPainter.h"
//#include "GpsPlayMailbox.h"
//#include "MSensorAtPortEmu.h"
#include "HumanHandImu.h"

class GplayControlListeningSocket;
class GplayControlSocket: public TextTCPSocket
{
public:
	GplayControlSocket(FileDescriptorPoller *poller, int sockfd,
			DeferredExecutor *deferredexecutor,
			TimeoutDispatcher *timeoutdispatcher,
			GplayControlListeningSocket *father);
	virtual ~GplayControlSocket();

	virtual void somethingWasRead()
	{
	}
	virtual void nothingToSend()
	{
	}
	virtual void stringReceived(const std::string &received);

	virtual void disconnectHappened(int fd);
	virtual void errorHappened(int fd);
	virtual void endOfFileEncounteredOnInput();

	virtual void binaryBlockReceived(unsigned char *block, size_t size)
	{
	}
protected:
	TimeoutDispatcher *timeoutdispatcher;
	GplayControlListeningSocket *father;

};

class GplayControlListeningSocket: public TCPListeningSocket
{
public:
	GplayControlListeningSocket(FileDescriptorPoller *poller,
			DeferredExecutor *deferredexecutor,
			TimeoutDispatcher *timeoutdispatcher, FrameTimer *frametimer,
			MeshReader *reader, TlEmu *tl_painter/*,
			GpsPlayMailbox *gsm_player, MSensorAtPortEmu *atport_emu*/,
			HumanHandImu *camera_control_local_imu,
			std::string historyfile);
	virtual ~GplayControlListeningSocket();
	
	//own interface
	bool openLastFromHistory();
	bool openFile(std::string rc_name, bool save_to_history);

	// TCPListeningSocket interface
	virtual void connectionAccepted(int acceptedsockfd,
			unsigned short int portfrom, u_int32_t ipfrom);

private:
	void addToHistory(std::string fname);

	friend class GplayControlSocket;
	void socketDisconnected(GplayControlSocket *sock);
public:
	void pause();
	void play();
	void nextBlock();
	std::string playerState();

	bool wanttoexit;

protected:
	DeferredExecutor *deferredexecutor;
	TimeoutDispatcher *timeoutdispatcher;
	FrameTimer *frametimer;
	MeshReader *reader;
	TlEmu *tl_painter;
//	GpsPlayMailbox *gsm_player;
//	MSensorAtPortEmu *atport_emu;
	HumanHandImu *camera_control_local_imu;

	std::string historyfile;

	std::set<GplayControlSocket*> connections;
};

#endif /* GPLAYCONTROLSOCKET_H_ */
