/*
 * GplayControlSocket.cpp
 *
 *      Author: bond
 */

#include "GplayControlSocket.h"
#include "FrameTimer.h"
#include "RegexpMatcher.h"
#include "PerfCounter.h"

#include <string.h>
#include <errno.h>
#include <sstream>

//==============================================================================
//===================== GplayControlSocket ================================
//==============================================================================
GplayControlSocket::GplayControlSocket(FileDescriptorPoller *poller, int sockfd,
		DeferredExecutor *deferredexecutor,
		TimeoutDispatcher *timeoutdispatcher, GplayControlListeningSocket *father)
		: TextTCPSocket(sockfd, deferredexecutor)
{
	this->poller = poller;
	this->timeoutdispatcher = timeoutdispatcher;
	this->father = father;

	addToPoller(poller);
	wantToReadString();
}

GplayControlSocket::~GplayControlSocket()
{
	poller->excludeConnection(sockfd);
	father->socketDisconnected(this);
}

void GplayControlSocket::disconnectHappened(int fd)
{
	delete this;
}

void GplayControlSocket::errorHappened(int fd)
{
	delete this;
}

void GplayControlSocket::endOfFileEncounteredOnInput()
{
	delete this;
}

void GplayControlSocket::stringReceived(const std::string &received)
{
//	std::cout << "GplayControlSocket::stringReceived: " << received << std::endl;
	static RegexpMatcher setFpsPattern("^SETFPS: ([0-9]+)$",
			RegexpMatcher::EXTENDED);

	static RegexpMatcher debugCaptureStartPattern(
			"^DEBUGCAPTURE START ([0-9-]+)$", RegexpMatcher::EXTENDED);
	static RegexpMatcher debugCaptureExStartPattern(
			"^DEBUGCAPTURE START ([0-9-]+) SAVEIMAGE=(NO|YES)$",
			RegexpMatcher::EXTENDED);
	static RegexpMatcher debugCaptureStopPattern("^DEBUGCAPTURE STOP$",
			RegexpMatcher::EXTENDED);
	static RegexpMatcher debugCaptureCommentPattern(
			"^DEBUGCAPTURE COMMENT: (.*)$", RegexpMatcher::EXTENDED);

	static RegexpMatcher setTlPainterConfigPattern(
			"^SET_TLPAINTER_CONFIG (.*)$", RegexpMatcher::EXTENDED);

	static RegexpMatcher setCameraLoockAtPattern(
			"^SET_CAM_LOOK_AT(.*)$", RegexpMatcher::EXTENDED);
	static RegexpMatcher setCameraDirectionPattern(
			"^SET_CAM_DIRECTION(.*)$", RegexpMatcher::EXTENDED);
	static RegexpMatcher getCameraDirectionPattern(
			"^GET_CAM_DIRECTION(.*)$", RegexpMatcher::EXTENDED);

	if (received == "p") {
		father->frametimer->pause();
	} else if (received == "pr") {
		;
	} else if (received == "play") {
		father->frametimer->play();
	} else if (received == "stop") {
		father->frametimer->pause();
		father->reader->toStart();
	} else if (received == "tostart") {
		father->frametimer->pause();
		father->reader->toStart();
		father->frametimer->play();
	} else if (received == "pause") {
		father->frametimer->pause();
	} else if (setFpsPattern.match(received)) {
		int fps = atoi(setFpsPattern.retrieveMatchedSubstring(1).c_str());
		printf("set play fps to %d\n", fps);
		father->frametimer->setFps(fps);
	} else if (debugCaptureStartPattern.match(received)) {
		father->frametimer->play();
	} else if (debugCaptureExStartPattern.match(received)) {
		father->frametimer->play();
	} else if (debugCaptureStopPattern.match(received)) {
		father->frametimer->pause();
	} else if (debugCaptureCommentPattern.match(received)) {
		std::string fname = debugCaptureCommentPattern.retrieveMatchedSubstring(
				1);
		father->openFile(fname, true);
	} else if (received == "DEBUGCAPTURESTATE") {
		std::ostringstream s;
		s << "DEBUGCAPTURESTATE " << father->frametimer->getStateStr()
				<< std::endl;
		sendString(s.str());
	} else if (received == "GET_TLPAINTER_CONFIG") {
		std::string tl_config;
		if (father->tl_painter) {
			tl_config = father->tl_painter->getConfig();
		}
		std::ostringstream s;
		s << "TLPAINTER_CONFIG " << tl_config << std::endl;
		sendString(s.str());
	} else if (setTlPainterConfigPattern.match(received)) {
		std::string tl_config =
				setTlPainterConfigPattern.retrieveMatchedSubstring(1);
		if (!father->tl_painter) {
			sendString("SET_TLPAINTER_CONFIG ERROR no rl_detector\n");
			wantToReadString();
			return;
		}
		if (!father->tl_painter->setConfig(tl_config)) {
			sendString("SET_TLPAINTER_CONFIG ERROR wrong config\n");
			wantToReadString();
			return;
		}
		sendString("SET_TLPAINTER_CONFIG OK\n");
	} else if (received == "GET_CAM_LOOK_AT") {
		std::shared_ptr<sutil::Camera> camera =
				father->reader->scene.cameraByName("Camera", nullptr);
		if (!camera) {
			sendString("GET_CAM_LOOK_AT ERR\n");
			wantToReadString();
			return;
		}
		float3 look_at = camera->lookat();
		ParamsStr p;
		p.setParamDouble("X", look_at.x);
		p.setParamDouble("Y", look_at.y);
		p.setParamDouble("Z", look_at.z);
		std::ostringstream s;
		s << "CAM_LOOK_AT " << p.toString() << std::endl;
		sendString(s.str());
		wantToReadString();
		return;
	} else if (setCameraLoockAtPattern.match(received)) {
		std::string params_str =
				setCameraLoockAtPattern.retrieveMatchedSubstring(1);
		std::shared_ptr<sutil::Camera> camera =
				father->reader->scene.cameraByName("Camera", nullptr);
		if (!camera) {
			sendString("SET_CAM_LOOK_AT ERR\n");
			wantToReadString();
			return;
		}
		ParamsStr p(params_str);
		float3 now_look_at = camera->lookat();
		float3 look_at;
		look_at.x = p.getParamDouble("X", now_look_at.x);
		look_at.y = p.getParamDouble("Y", now_look_at.y);
		look_at.z = p.getParamDouble("Z", now_look_at.z);
		camera->setLookatNotChangeFocalLength(look_at);
		sendString("SET_CAM_LOOK_AT OK\n");
		wantToReadString();
		return;
	} else if (getCameraDirectionPattern.match(received)) {
		std::string params_str =
				getCameraDirectionPattern.retrieveMatchedSubstring(1);
		ParamsStr p_in(params_str);
		std::shared_ptr<sutil::Camera> camera =
				father->reader->scene.cameraByName(p_in.getParam("CAMERA"));
		if (!camera) {
			sendString("GET_CAM_DIRECTION ERR: no camera\n");
			wantToReadString();
			return;
		}
		float3 dir = camera->direction();
		ParamsStr p;
		p.setParamDouble("X", dir.x);
		p.setParamDouble("Y", dir.y);
		p.setParamDouble("Z", dir.z);
		std::ostringstream s;
		s << "CAM_DIRECTION " << p.toString() << std::endl;
		sendString(s.str());
		wantToReadString();
		return;
	} else if (setCameraDirectionPattern.match(received)) {
		std::string params_str =
				setCameraDirectionPattern.retrieveMatchedSubstring(1);
		ParamsStr p(params_str);
		std::shared_ptr<sutil::Camera> camera =
				father->reader->scene.cameraByName(p.getParam("CAMERA"));
		if (!camera) {
			sendString("SET_CAM_DIRECTION ERR: no camera\n");
			wantToReadString();
			return;
		}
		if (p.hasParam("LAT") || p.hasParam("LON")) {
			//SET_CAM_DIRECTION LAT=-2 LON=90
			float m_latitude = p.getParamDouble("LAT", 0.0) * M_PI / 180.0;
			float m_longitude = p.getParamDouble("LON", 0.0) * M_PI / 180.0;
			float3 localDir;
			localDir.x = cos(m_latitude) * sin(m_longitude);
			localDir.y = cos(m_latitude) * cos(m_longitude);
			localDir.z = sin(m_latitude);
			float3 u, v, w;
			//camera->UVWFrame(u,v,w);
			u = make_float3(1.0f, 0.0f, 0.0f);
			v = make_float3(0.0f, 0.0f, 1.0f);
			w = make_float3(0.0f, 1.0f, 0.0f);
			float3 dirWS = u * localDir.x + v * localDir.y + w * localDir.z;
			const float3 &eye = camera->eye();
			float m_cameraEyeLookatDistance =
					length(camera->lookat() - camera->eye());
			camera->setLookat(eye - dirWS * m_cameraEyeLookatDistance);
			if (father->camera_control_local_imu)
				father->camera_control_local_imu->disable();
		} else {
			float3 now_dir = camera->direction();
			float3 dir;
			dir.x = p.getParamDouble("X", now_dir.x);
			dir.y = p.getParamDouble("Y", now_dir.y);
			dir.z = p.getParamDouble("Z", now_dir.z);
			camera->setDirection(normalize(dir));
		}
		sendString("SET_CAM_DIRECTION OK\n");
		wantToReadString();
		return;
	} else if (received == "ENABLEIMU") {
		sendString("ENABLEIMU OK\n");
		if (father->camera_control_local_imu) {
			father->camera_control_local_imu->enable();
			father->camera_control_local_imu->startSelfCalib();
		}
		wantToReadString();
		return;
	} else if (received == "EXITNOW") {
		father->wanttoexit = true;
		std::cout << "Exit by EXITNOW" << std::endl;
	} else {
		father->nextBlock();
	}
	wantToReadString();
}

//==============================================================================
//===================== GplayControlListeningSocket ================================
//==============================================================================
GplayControlListeningSocket::GplayControlListeningSocket(
		FileDescriptorPoller *poller, DeferredExecutor *deferredexecutor,
		TimeoutDispatcher *timeoutdispatcher, FrameTimer *frametimer,
		MeshReader *reader, TlEmu *tl_painter/*,
		GpsPlayMailbox *gsm_player, MSensorAtPortEmu *atport_emu*/,
		HumanHandImu *camera_control_local_imu,
		std::string historyfile)
{
	this->deferredexecutor = deferredexecutor;
	this->timeoutdispatcher = timeoutdispatcher;
	this->frametimer = frametimer;
	this->reader = reader;
	this->tl_painter = tl_painter;
//	this->gsm_player = gsm_player;
//	this->atport_emu = atport_emu;
	this->camera_control_local_imu = camera_control_local_imu;

	this->historyfile = historyfile;

	wanttoexit = false;
}

GplayControlListeningSocket::~GplayControlListeningSocket()
{
	std::set<GplayControlSocket*> connections_copy = connections;
	for (std::set<GplayControlSocket*>::iterator it = connections_copy.begin();
			it != connections_copy.end(); it++)
	{
		delete *it;
	}
}

void GplayControlListeningSocket::socketDisconnected(GplayControlSocket *sock)
{
	connections.erase(sock);
}

void GplayControlListeningSocket::connectionAccepted(int acceptedsockfd,
		unsigned short int portfrom, u_int32_t ipfrom)
{
	GplayControlSocket *conn = new GplayControlSocket(poller, acceptedsockfd,
			deferredexecutor, timeoutdispatcher, this);

	struct in_addr addr;
	addr.s_addr = htonl(ipfrom);
	std::cout << "New PlayContro connection from " << inet_ntoa(addr) << ":"
			<< portfrom << std::endl;
	connections.insert(conn);
}

void GplayControlListeningSocket::nextBlock()
{
	reader->nextFrame(true);
}

void GplayControlListeningSocket::addToHistory(std::string fname)
{
	if (historyfile == "")
		return;
	FILE *fp = fopen(historyfile.c_str(), "wt");
	if (!fp) {
		printf("can't save history (can't create %s : %s)\n",
				historyfile.c_str(), strerror(errno));
		return;
	}
	fprintf(fp, "%s", fname.c_str());
	fclose(fp);
	printf("save to history: %s\n", fname.c_str());
}

static std::vector<uint8_t> loadFileToBuf(std::string filename)
{
	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp) {
		return std::vector<uint8_t>();
	}
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	std::vector<uint8_t> buf(len);
	int ret = fread(buf.data(), 1, len, fp);
	if (ret != len) {
		return std::vector<uint8_t>();
	}
	fclose(fp);
	return buf;
}

bool GplayControlListeningSocket::openFile(std::string rc_name, bool save_to_history)
{
	bool ret = reader->openFile(rc_name);
	if (!ret)
		return false;
//	if (gsm_player)
//		gsm_player->openFile(rc_name + ".nmea");
	if (tl_painter) {
		std::vector<uint8_t> b = loadFileToBuf(rc_name + ".tl");
		std::string tl_conf((const char*)b.data(), b.size());
		tl_painter->setConfig(tl_conf);
	}
//	if (atport_emu) {
//		atport_emu->setAtvString("");
//		atport_emu->setZoom(0.0);
//	}
	if (save_to_history)
		addToHistory(rc_name);
	return true;
}

bool GplayControlListeningSocket::openLastFromHistory()
{
	if (historyfile == "")
		return false;
	FILE *fp = fopen(historyfile.c_str(), "rt");
	if (!fp) {
		printf("can't load history (can't open %s : %s)\n", historyfile.c_str(),
				strerror(errno));
		return false;
	}
	char buf[1024];
	int ret = fscanf(fp, "%s", buf);
	fclose(fp);
	if (ret != 1) {
		printf("wrong or empty histore record (file %s : %s)\n",
				historyfile.c_str(), strerror(errno));
		return false;
	}
	std::string rc_name = buf;
	return openFile(buf, false);
}
