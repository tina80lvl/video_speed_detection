/*
 * GplayWebDisplay.cpp
 *
 *  Created on: 24.04.2020
 *      Author: bond
 */

#include "GplayWebDisplay.h"

#ifdef HAVE_WITTY
#include <algorithm>
#include <Wt/Http/Request.h>
#include <Wt/Http/Response.h>
#include <Wt/WPanel.h>
#include <Wt/WSlider.h>

#include "AllocatedFrameDescriptor.h"
#include "GrabbedFrameDescriptorWrapper.h"
#include "PerfCounter.h"

using boost::asio::ip::tcp;

//==============================================================================
//===================== FilesTableModel ========================================
//==============================================================================

FilesTableModel::FilesTableModel(const std::string &topdir):
		WStringListModel(), topdir(topdir)
{
}

void FilesTableModel::load()
{
	if (files.size()) {
		beginRemoveRows(WModelIndex(), 0, files.size() - 1);
		files.clear();
		endRemoveRows();
	}

	boost::filesystem::path top_path(topdir);
	boost::filesystem::recursive_directory_iterator end_iter;
	for (boost::filesystem::recursive_directory_iterator dir_itr(top_path);
			dir_itr != end_iter; ++dir_itr) {
		if (!boost::filesystem::is_regular_file(dir_itr->status()))
			continue;
		boost::filesystem::path fpath = dir_itr->path();
//		if (fpath.extension() != ".glb" || fpath.extension() != ".gltf")
//			continue;
		files.push_back(fpath);
	}
	std::sort<std::vector<boost::filesystem::path>::iterator>(files.begin(),
			files.end());

	if (files.size()) {
		beginInsertRows(WModelIndex(), 0, files.size() - 1);
		endInsertRows();
	}
}

int FilesTableModel::rowCount(const WModelIndex &parent) const
{
	return files.size();
}

cpp17::any
FilesTableModel::data(const WModelIndex &index, ItemDataRole role) const
{
	if (role != ItemDataRole::Display)
		return cpp17::any();
	if (index.column() != 0)
		return cpp17::any();
	if (index.row() >= (int) files.size())
		return cpp17::any();

	return files[index.row()].string();
}

//==============================================================================
//===================== VideoFrameQueueResource ================================
//==============================================================================
VideoFrameQueueResource::VideoFrameQueueResource():
		WResource()
{
	currentframe = 0;
	suggestFileName("frame.jpg");
}

VideoFrameQueueResource::~VideoFrameQueueResource()
{
	beingDeleted(); // see "Concurrency issues" for WResource
	if (currentframe)
		currentframe->releaseObject("VideoFrameQueueResource");
}

void VideoFrameQueueResource::setFrame(FrameDescriptor *frame)
{
	if (currentframe)
		currentframe->releaseObject("VideoFrameQueueResource");
	currentframe = frame;
	if (currentframe)
		currentframe->claimObject("VideoFrameQueueResource");
	setChanged();
}

void VideoFrameQueueResource::handleRequest(const Wt::Http::Request &request,
		Wt::Http::Response &response)
{
	if (!currentframe) {
		response.setStatus(404);
		return;
	}
	response.setMimeType("image/jpeg");
	response.out().write((const char *) currentframe->getContents(),
			currentframe->getLength());
}

int VideoFrameQueueResource::width()
{
	if (!currentframe)
		return 0;
	return currentframe->width;
}

int VideoFrameQueueResource::height()
{
	if (!currentframe)
		return 0;
	return currentframe->height;
}

//==============================================================================
//===================== GplayCameraWidget ========================================
//==============================================================================
GplayCameraWidget::GplayCameraWidget(int def_w)
{
	vq = new VideoFrameQueueResource();

	int w = def_w;
	int h = w * 1080 / 1920;
	resize(w, h); // default size
}

GplayCameraWidget::~GplayCameraWidget()
{
	delete vq;
}

void GplayCameraWidget::paintEvent(WPaintDevice *paintDevice)
{
	WPainter painter(paintDevice);
	painter.drawImage(WRectF(0, 0, width().toPixels(), height().toPixels()),
			WPainter::Image(vq->url(), vq->width(), vq->height()));
}

void GplayCameraWidget::setFrame(FrameDescriptor *jpeg_frame)
{
	int now_w = width().toPixels();
	int now_h = height().toPixels();
	if (now_w * jpeg_frame->height != now_h * jpeg_frame->width) {
		// ratio changed
		int new_h = now_w * jpeg_frame->height / jpeg_frame->width;
		resize(now_w, new_h);
	}
	vq->setFrame(jpeg_frame);
	update();
}

void GplayCameraWidget::setDw(int dw)
{
	if (!vq->width())
		return;
	int dh = dw * vq->height() / vq->width();
	resize(dw, dh);
}

//==============================================================================
//===================== GplaySiteGui ================================================
//==============================================================================
void setSliderProps(WSlider * angle_slider, float min_v, float max_v, float tick, float val, float mul)
{
	angle_slider->setMinimum(min_v*mul);
	angle_slider->setMaximum(max_v*mul);
	angle_slider->setValue(val*mul);
	angle_slider->setTickInterval(tick*mul);
	angle_slider->setTickPosition(Wt::WSlider::TicksBothSides);
}

GplaySiteGui::GplaySiteGui(GplayWebDisplay *wd): WContainerWidget()
{
	this->wd = wd;
	this->m_traking = false;

	filestablemodel = std::make_shared<FilesTableModel>(wd->recordsdir);
	fileselectionbox = 0;

	WPanel *play_info_panel = addWidget(std::make_unique<Wt::WPanel>());
	play_info_panel->setTitle("Play info");
	WContainerWidget *play_info_container = play_info_panel->setCentralWidget(
			std::make_unique<Wt::WContainerWidget>());
	frame_time = play_info_container->addWidget(std::make_unique<WText>("-1"));
	play_info_container->addWidget(std::make_unique<WText>(" frameidx: "));
	frame_number = play_info_container->addWidget(std::make_unique<WText>(""));
	play_info_container->addWidget(std::make_unique<WText>(" frameticks: "));
	frame_ticks = play_info_container->addWidget(std::make_unique<WText>(""));

	WPanel *play_control_panel = addWidget(std::make_unique<Wt::WPanel>());
	play_control_panel->setTitle("Play control");
	WContainerWidget *play_control_container =
			play_control_panel->setCentralWidget(
					std::make_unique<Wt::WContainerWidget>());

	WPanel *camera_control_panel = addWidget(std::make_unique<Wt::WPanel>());
	camera_control_panel->setTitle("Camera view control");
	WContainerWidget *camera_control_container =
			camera_control_panel->setCentralWidget(
					std::make_unique<Wt::WContainerWidget>());

	// slider for camera widget size control
	WSlider *camera_scale_slider = camera_control_container->addWidget(
			std::make_unique<Wt::WSlider>(Wt::Orientation::Horizontal));
	camera_scale_slider->setMinimum(10);
	camera_scale_slider->setMaximum(1920);
	camera_scale_slider->setValue(360);
	camera_scale_slider->setTickInterval(100);
	camera_scale_slider->setTickPosition(Wt::WSlider::TicksBothSides);
	camera_scale_slider->resize(300, 50);

	camera_widget = addWidget(
			std::make_unique<GplayCameraWidget>(camera_scale_slider->value()));
	camera_widget->setStyleClass("cameraview");
	camera_scale_slider->valueChanged().connect(camera_widget,
			&GplayCameraWidget::setDw);

	if (wd->controlport) {
		WPushButton *fileselect_button = play_control_container->addWidget(
				std::make_unique<Wt::WPushButton>("select file"));
		WPushButton *play_button = play_control_container->addWidget(
				std::make_unique<Wt::WPushButton>("play"));
		WPushButton *pause_button = play_control_container->addWidget(
				std::make_unique<Wt::WPushButton>("pause"));
		WPushButton *nextfr_button = play_control_container->addWidget(
				std::make_unique<Wt::WPushButton>("next fr"));

		fileselect_button->clicked().connect(this,
				&GplaySiteGui::selectFilePressed);
		play_button->clicked().connect(this, &GplaySiteGui::playPressed);
		pause_button->clicked().connect(this, &GplaySiteGui::pausePressed);
		nextfr_button->clicked().connect(this, &GplaySiteGui::nextFrPressed);

		char buf[256];
		sprintf(buf, "%d", wd->controlport);
		tcp::resolver::query query("127.0.0.1", buf); // "33333"
		tcp::resolver resolver(io_service);
		tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
		controlsock = new tcp::socket(io_service);
		controlsock->connect(*endpoint_iterator);

		cam_lat = camera_control_container->addWidget(
				std::make_unique<Wt::WSlider>(Wt::Orientation::Horizontal));
		setSliderProps(cam_lat, -3.0, 3.0, 0.2, 0.0, 10);
		cam_lon = camera_control_container->addWidget(
				std::make_unique<Wt::WSlider>(Wt::Orientation::Horizontal));
//		setSliderProps(cam_lon, -5.0, 5.0, 0.2, 0.0, 10);
		setSliderProps(cam_lon, 90.0-5.0, 90.0+5.0, 0.2, 90.0, 10);

//		cam_lat->valueChanged().connect(this,
//				&GplaySiteGui::camLatLonChanged);
//		cam_lon->valueChanged().connect(this,
//				&GplaySiteGui::camLatLonChanged);

		camera_widget->mouseWentDown().connect(this,
				&GplaySiteGui::startTracking);
		camera_widget->mouseWentOut().connect(this,
				&GplaySiteGui::stopTracking);
		camera_widget->mouseWentUp().connect(this,
				&GplaySiteGui::stopTracking);
		camera_widget->mouseMoved().connect(this,
				&GplaySiteGui::updateTracking);

	} else {
		controlsock = 0;
	}

	wd->registerConnection(WApplication::instance()->sessionId(), this,
			boost::bind(&GplaySiteGui::updateData, this));

	Wt::WApplication::instance()->enableUpdates(true);
}

GplaySiteGui::~GplaySiteGui()
{
	if (controlsock)
		delete controlsock;
	wd->unRegisterConnection(this);
	Wt::WApplication::instance()->enableUpdates(false);
}

void GplaySiteGui::startTracking(Wt::WMouseEvent ev)
{
	m_traking = true;
	m_prevPosX = ev.widget().x;
	m_prevPosY = ev.widget().y;
}

void GplaySiteGui::stopTracking(Wt::WMouseEvent ev)
{
	m_traking = false;
}

void GplaySiteGui::updateTracking(Wt::WMouseEvent ev)
{
	if (!m_traking)
		return;

	int deltaX = ev.widget().x - m_prevPosX;
	int deltaY = ev.widget().y - m_prevPosY;
	m_prevPosX = ev.widget().x;
	m_prevPosY = ev.widget().y;

	float latitude = (float) cam_lat->value() / 10.0;
	float longitude = (float) cam_lon->value() / 10.0;
	latitude = std::min(89.0f, std::max(-89.0f, latitude + 0.05f * deltaY));
	longitude = fmod(longitude - 0.05f * deltaX, 360.0f);
	cam_lat->setValue(latitude * 10.0);
	cam_lon->setValue(longitude * 10.0);
	camLatLonChanged(0);
}

void GplaySiteGui::updateData()
{
	GrabbedFrameDescriptor *frame = 0;
	{
		boost::mutex::scoped_lock lock(q_mutex);

		GrabbedFrameDescriptor *prev = 0;
		while (!framestoplay.empty()) {
			frame = framestoplay.front();
			framestoplay.pop();
			if (prev) {
				framesfromplay.push(prev);
			}
			prev = frame;
		}
		if (!frame) {
			return;
		}
	}

	// frame params
	frame_number->setText(WString("{1}").arg((int) frame->frameidx));
	char buf[256];
	sprintf(buf, "%s (%u.%06u)",
			getDateTimeStringRepresentation2(frame->capturingtime).c_str(),
			(unsigned) frame->capturingtime.tv_sec,
			(unsigned) frame->capturingtime.tv_usec);
	frame_time->setText(WString("{1}").arg(buf));
	sprintf(buf, "%u",frame->frameticks);
	frame_ticks->setText(WString("{1}").arg(buf));
	camera_widget->setFrame(frame);
	{
		boost::mutex::scoped_lock lock(q_mutex);
		framesfromplay.push(frame);
	}
	Wt::WApplication::instance()->triggerUpdate();
}

void GplaySiteGui::playPressed()
{
	controlsock->send(boost::asio::buffer("play\n"));
}

void GplaySiteGui::pausePressed()
{
	controlsock->send(boost::asio::buffer("p\n"));
}

void GplaySiteGui::nextFrPressed()
{
	controlsock->send(boost::asio::buffer("\n"));
}

void GplaySiteGui::selectFilePressed()
{
	Wt::WDialog
			*dialog = addChild(std::make_unique<Wt::WDialog>("File Selection"));
	filestablemodel->load();

	dialog->contents()->addWidget(std::make_unique<Wt::WText>(filestablemodel->topdir));
	dialog->contents()->addWidget(std::make_unique<Wt::WBreak>());
	fileselectionbox = dialog->contents()->addWidget(
			std::make_unique<Wt::WSelectionBox>());
	fileselectionbox->setModel(filestablemodel);

	dialog->contents()->addWidget(std::make_unique<Wt::WBreak>());
	Wt::WPushButton *ok = dialog->contents()->addWidget(
			std::make_unique<Wt::WPushButton>("Ok"));
	ok->clicked().connect(dialog, &Wt::WDialog::accept);
	Wt::WPushButton *cancel = dialog->contents()->addWidget(
			std::make_unique<Wt::WPushButton>("Cancel"));
	cancel->clicked().connect(dialog, &Wt::WDialog::reject);

	dialog->finished().connect(this, &GplaySiteGui::selectFileFinished);
	dialog->show();
}

void GplaySiteGui::selectFileFinished(Wt::DialogCode r)
{
	if (r != Wt::DialogCode::Accepted) {
		return;
	}
	int idx = fileselectionbox->currentIndex();
	if (idx < 0 )
		return;
	boost::filesystem::path filep = filestablemodel->files[idx];
	printf("selected: %s\n", filep.string().c_str());

	std::ostringstream s;
	s << "DEBUGCAPTURE COMMENT: " << filep.string() << std::endl;
	controlsock->send(boost::asio::buffer(s.str()));
}

void GplaySiteGui::camLatLonChanged(int v)
{
	{
		ParamsStr p;
		p.setParam("CAMERA", "Camera");
		p.setParamDouble("LAT", (double) cam_lat->value() / 10.0); // mul is 10
		p.setParamDouble("LON", (double) cam_lon->value() / 10.0); // mul is 10
		std::ostringstream s;
		s << "SET_CAM_DIRECTION " << p.toString() << std::endl;
		controlsock->send(boost::asio::buffer(s.str()));
	}
	{
		ParamsStr p;
		p.setParam("CAMERA", "CameraDepth");
		p.setParamDouble("LAT", (double) cam_lat->value() / 10.0); // mul is 10
		p.setParamDouble("LON", (double) cam_lon->value() / 10.0); // mul is 10
		std::ostringstream s;
		s << "SET_CAM_DIRECTION " << p.toString() << std::endl;
		controlsock->send(boost::asio::buffer(s.str()));
	}
}

//==============================================================================
//===================== GplayWebDisplay ================================
//==============================================================================
std::unique_ptr<WApplication>
createApplication(const WEnvironment &env, GplayWebDisplay *wd)
{
	std::unique_ptr<WApplication> app = std::make_unique<Wt::WApplication>(env);
	app->setCssTheme("polished");
	app->setTitle("Gplay records view");
	app->root()->addWidget(std::make_unique<GplaySiteGui>(wd));
	return app;
}

GplayWebDisplay::GplayWebDisplay( int webport, unsigned short controlport,
		std::string recordsdir, ImageConversionQueue *image_q_control)
{
	this->controlport = controlport;
	this->recordsdir = recordsdir;
	this->image_q_control = image_q_control;

	server = new WServer("GplayWebDisplay");

//	const char *argv[] = { "web", "--docroot", "/usr/share/Wt",
//			"--http-address", "0.0.0.0", "--http-port", "8080" };
	const char *argv[] =
			{ "web", "--docroot", "/usr/share/Wt", "--http-address",
			  "0.0.0.0", "--http-port", "8080" };

	char portbuf[64];
	sprintf(portbuf, "%d", webport);
	argv[6] = portbuf;

	server->setServerConfiguration(7, (char **) argv);
	server->addEntryPoint(Wt::EntryPointType::Application,
			boost::bind(createApplication, _1, this));
	if (!server->start()) {
		printf("can't start WServer\n");
		exit(1);
	}
	// now WServer run in separate thread
}

GplayWebDisplay::~GplayWebDisplay()
{
	for (unsigned i = 0; i < connections.size(); ++i) {
		Connection &c = connections[i];
		boost::mutex::scoped_lock lock(c.client->q_mutex);
		while (!c.client->framesfromplay.empty()) {
			FrameDescriptor *fdone = c.client->framesfromplay.front();
			c.client->framesfromplay.pop();
			fdone->releaseObject();
		}
		while (!c.client->framestoplay.empty()) {
			FrameDescriptor *fcancel = c.client->framestoplay.front();
			c.client->framestoplay.pop();
			fcancel->releaseObject();
		}
	}
	server->stop();
	delete server;
	server = 0;
}

void GplayWebDisplay::frameGrabbed(GrabbedFrameDescriptor *frame)
{
	if (!frame->isClaimAllowed()) {
		return;
	}
	for (unsigned i = 0; i < connections.size(); ++i) {
		Connection &c = connections[i];
		boost::mutex::scoped_lock lock(c.client->q_mutex);
		if (c.bufs_use < 2) {
			frame->claimObject("GplayWebDisplay");
			PerfCounter::incrementTotalCounter("claimed_by_web_gui", 1);
			c.client->framestoplay.push(frame);
			c.bufs_use++;
			server->post(c.sessionid, c.function);
		}
		while (!c.client->framesfromplay.empty()) {
			FrameDescriptor *fdone = c.client->framesfromplay.front();
			c.client->framesfromplay.pop();
			fdone->releaseObject("GplayWebDisplay");
			PerfCounter::incrementTotalCounter("claimed_by_web_gui", -1);
			c.bufs_use--;
		}
	}
}

void
GplayWebDisplay::registerConnection(std::string session, GplaySiteGui *client,
		const boost::function<void()> &notifyfunction)
{
	boost::mutex::scoped_lock lock(connections_mutex);
	connections.push_back(Connection(session, client, notifyfunction));
	if (!image_q_control->isWorking())
		image_q_control->startWorking();
}

void GplayWebDisplay::unRegisterConnection(GplaySiteGui *client)
{
	boost::mutex::scoped_lock lock(connections_mutex);
	for (unsigned i = 0; i < connections.size(); ++i) {
		if (connections[i].client != client)
			continue;
		connections.erase(connections.begin() + i);
		while (!client->framesfromplay.empty()) {
			FrameDescriptor *fdone = client->framesfromplay.front();
			client->framesfromplay.pop();
			fdone->releaseObject("GplayWebDisplay");
		}
		while (!client->framestoplay.empty()) {
			FrameDescriptor *fcancel = client->framestoplay.front();
			client->framestoplay.pop();
			fcancel->releaseObject("GplayWebDisplay");
		}
		break;
	}
	if (connections.empty())
		image_q_control->stopWorking();
}

#endif /* HAVE_WITTY */
