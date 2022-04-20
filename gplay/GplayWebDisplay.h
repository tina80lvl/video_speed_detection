/*
 * GplayWebDisplay.h
 *
 *  Created on: 24.04.2020
 *      Author: bond
 */

#ifndef GPLAYWEBDISPLAY_H_
#define GPLAYWEBDISPLAY_H_

#ifdef HAVE_WITTY

#include "ImageFormatConverter.h"
#include "VideoFrameQueue.h"
#include "GrabbedFrameDescriptor.h"
#include "ImageFormatConverter.h"

#include <Wt/WApplication.h>
#include <Wt/WServer.h>
#include <Wt/WContainerWidget.h>
#include <Wt/WText.h>
#include <Wt/WResource.h>
#include <Wt/WPushButton.h>
#include <Wt/WPaintedWidget.h>
#include <Wt/WPainter.h>
#include <Wt/WSelectionBox.h>
#include <Wt/WStringListModel.h>
#include <Wt/WDialog.h>

#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/filesystem/convenience.hpp>

#include <vector>

using boost::asio::ip::tcp;
using namespace Wt;

//==============================================================================
//===================== FilesTableModel ========================================
//==============================================================================
class FilesTableModel: public WStringListModel
{
public:
	FilesTableModel(const std::string &topdir);

	void load();

	int rowCount(const WModelIndex &parent = WModelIndex()) const;
	cpp17::any data(const WModelIndex &index,
			ItemDataRole role = ItemDataRole::Display) const;
public:
	std::string topdir;
	std::vector<boost::filesystem::path> files;
};

//==============================================================================
//===================== VideoFrameQueueResource ================================
//==============================================================================
class VideoFrameQueueResource: public WResource
{
public:
	VideoFrameQueueResource();
	virtual ~VideoFrameQueueResource();
	void setFrame(FrameDescriptor *frame);
	void handleRequest(const Wt::Http::Request& request,
			Wt::Http::Response& response);
	int width();
	int height();
private:
	FrameDescriptor *currentframe;
};

//==============================================================================
//===================== GplayCameraWidget ========================================
//==============================================================================
class GplayCameraWidget: public WPaintedWidget
{
public:
	GplayCameraWidget(int def_w);
	virtual ~GplayCameraWidget();
	void setFrame(FrameDescriptor *origframe);
	void setDw(int dw);
protected:
	void paintEvent(WPaintDevice *paintDevice);
private:
	VideoFrameQueueResource *vq;
};

//==============================================================================
//===================== GplaySiteGui ================================================
//==============================================================================
class GplayWebDisplay;
class GplaySiteGui: public WContainerWidget
{
public:
	GplaySiteGui(GplayWebDisplay *wd);
	virtual ~GplaySiteGui();

private:
	void updateData();

	void playPressed();
	void pausePressed();
	void nextFrPressed();

	void selectFilePressed();
	void selectFileFinished(Wt::DialogCode r);

	void camLatLonChanged(int v);

	void startTracking(Wt::WMouseEvent ev);
	void stopTracking(Wt::WMouseEvent ev);
	void updateTracking(Wt::WMouseEvent ev);

private:
	GplayWebDisplay *wd;

	WText *frame_number;
	WText *frame_time;
	WText *frame_ticks;

	boost::asio::io_service io_service;
	tcp::socket *controlsock;

	GplayCameraWidget *camera_widget;
	std::shared_ptr<FilesTableModel> filestablemodel;
	Wt::WSelectionBox *fileselectionbox;

	WSlider *cam_lat;
	WSlider *cam_lon;

	bool m_traking;
	int m_prevPosX;
	int m_prevPosY;

public:
	mutable boost::mutex q_mutex;
	std::queue<GrabbedFrameDescriptor *> framestoplay;
	std::queue<GrabbedFrameDescriptor *> framesfromplay;
};

//==============================================================================
//===================== GplayWebDisplay ================================================
//==============================================================================

class GplayWebDisplay: public GrabberFrameListener
{
public:
	GplayWebDisplay(int webport,
			unsigned short controlport, std::string recordsdir,
			ImageConversionQueue *image_q_control);
	virtual ~GplayWebDisplay();

	void registerConnection(std::string session, GplaySiteGui *client,
			const boost::function<void()>& notifyfunction);
	void unRegisterConnection(GplaySiteGui *client);

	//OutgoingFrameListener
	void frameGrabbed(GrabbedFrameDescriptor *frame);

public:
	unsigned short controlport;
	std::string recordsdir;

private:
	ImageConversionQueue *image_q_control;
	WServer *server;

	struct Connection
	{
		Connection(const std::string& sessionid, GplaySiteGui *client,
				const boost::function<void()>& function)
		{
			this->sessionid = sessionid;
			this->client = client;
			this->function = function;
			this->bufs_use = 0;
		}
		std::string sessionid;
		GplaySiteGui *client;
		boost::function<void()> function;
		int bufs_use;
	};
	std::vector<Connection> connections;
	mutable boost::mutex connections_mutex;
};

#endif /* HAVE_WITTY */

#endif /* GPLAYWEBDISPLAY_H_ */
