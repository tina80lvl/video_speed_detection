#include "PureLiveVideoListeningSocket.h"

#include "RegexpMatcher.h"
#include "TimeUtilities.h"
#include "PerfCounter.h"
#include "ICONVConverter.h"

#include "math_utils.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <sstream>
#include <chrono>

#include <sutil/Quaternion.h>

using namespace std;

static auto start = chrono::high_resolution_clock::now();

static const std::string BOUNDARY = "boundarydonotcross";

//=========================================================================== 
//========================= SnapshotWaiter ==================================
//===========================================================================
PureLiveVideoSocket::SnapshotWaiter::SnapshotWaiter(
		PureLiveVideoSocket *return_to, VideoFrameQueue *framequeue,
		ImageFormatConverter *imageformatconverter)
{
	this->return_to = return_to;
	this->framequeue = framequeue;
	this->imageformatconverter = imageformatconverter;

	compressid = -1;
	jpegquality = 0;
	current_frame_arleady_sended = false;
	frame_on_shift_needed = false;
	targerframe = 0;

	framequeue->addQueueListener(this, targerframe, targerframe);
}

PureLiveVideoSocket::SnapshotWaiter::~SnapshotWaiter()
{
	framequeue->delQueueListener(this);
	if (compressid != -1)
		imageformatconverter->cancelConversion(compressid);
}

void PureLiveVideoSocket::SnapshotWaiter::request(int jpegquality,
		std::string snapshotsizename)
{
	this->jpegquality = jpegquality;
	this->snapshotsizename = snapshotsizename;

	if (current_frame_arleady_sended
			|| targerframe >= framequeue->getQueueLength()) {
		// ждем сдвига очереди
		frame_on_shift_needed = true;
		return;
	} else {
		sendFrame(framequeue->getFrameByNum(targerframe));
		current_frame_arleady_sended = true;
	}
}

void PureLiveVideoSocket::SnapshotWaiter::frameQueueShifted(
		VideoFrameQueue *framequeue)
{
	if (targerframe >= framequeue->getQueueLength())
		return;
	if (frame_on_shift_needed) {
		sendFrame(framequeue->getFrameByNum(targerframe));
		frame_on_shift_needed = false;
		current_frame_arleady_sended = true;
	} else {
		current_frame_arleady_sended = false;
	}
}

void
PureLiveVideoSocket::SnapshotWaiter::sendFrame(GrabbedFrameDescriptor *frame)
{
//	conversionFinished(0, frame);
//	return;
	if (snapshotsizename != "") { // !compatible
		report = std::make_pair("", 0);
	}
	// Оправить на конверсию, когда вернется - запишется в сокет
	std::ostringstream dest_frameformat;
	if (jpegquality == 0) {
		dest_frameformat << frame->getFrameFormatDescription();
		compressid = imageformatconverter->startConversion(frame,
				dest_frameformat.str(), 720, 576, this);
	} else {
//        dest_frameformat << "GRAYSCALE";
		dest_frameformat << snapshotsizename << "JPEG:" << jpegquality;
		compressid = imageformatconverter->startConversion(frame,
				dest_frameformat.str(), this);
	}
}

void PureLiveVideoSocket::SnapshotWaiter::conversionFinished(int,
		FrameDescriptor *converted)
{
	compressid = -1;
	size_t imagelen = converted ? converted->getLength() : 0;
	std::ostringstream s;

	if (http_) {
		s << "HTTP/1.1 200 OK\r\n" << "Connection: close\r\n"
				<< "Content-Type: image/jpeg\r\n" << "Content-Length: "
				<< imagelen << "\r\n\r\n";
	} else {
		if (snapshotsizename == "") { // compatible
			s << "SNAPSHOT LEN=" << imagelen << " CQ=" << jpegquality;
		} else {
			std::string str;
			switch (report.second) {
				default:
				case 0:
					str = "NONE";
					break;
				case 1:
					str = "SMALL";
					break;
				case 2:
					str = "NORMAL";
					break;
				case 3:
					str = "LARGE";
					break;
			}
			s << snapshotsizename << "SNAPSHOT LEN=" << imagelen << " CQ="
					<< jpegquality << " OPPOSITE=" << 0 << " FORWARD=" << 0
					<< " OTHER_OPPOSITE=" << 0 << " OTHER_FORWARD=" << 0
					<< " LICNUMBERSIZE_OK=" << str << " LICNUMBER="
					<< report.first;
		}
		s << " FORMAT=" << converted->getFrameFormatDescription() << "\n";
	}

	return_to->sendString(s.str());
	// Отправить тело сконвертированной картинки
	if (imagelen > 0) {
		BlockWithFrame *blocKwithframe = new BlockWithFrame(converted, 0);
		return_to->sendBlock(blocKwithframe);
		blocKwithframe->release();
	}

	// Приступаем к дальнейшему получению строк
	return_to->wantToReadString();
}

//=========================================================================== 
//========================= PureLiveVideoSocket =================================
//===========================================================================
PureLiveVideoSocket::PureLiveVideoSocket(int sockfd,
		PureLiveVideoListeningSocket *father, FileDescriptorPoller *poller,
		DeferredExecutor *deferredexecutor,
		ImageFormatConverter *imageformatconverter, VideoFrameQueue *framequeue,
		MeshReader *reader):
		TextTCPSocket(sockfd, deferredexecutor), father(father),
		imageformatconverter(imageformatconverter), framequeue(framequeue),
		snapshotSendder(0), wasAuthenticated(false), mesh_reader_(reader)
{
	addToPoller(poller);
	wantToReadString();
}

PureLiveVideoSocket::~PureLiveVideoSocket()
{
	delete snapshotSendder;
	father->connectionClosed(this);
	delete mjpeg_;
}

std::vector<std::string>
splitString(const std::string &str, const std::string &separator)
{
	std::vector<std::string> result;
	size_t prev = 0;
	size_t pos = str.find(separator);
	while (pos != std::string::npos) {
		result.push_back(str.substr(prev, pos - prev));
		prev = pos + 1;
		pos = str.find(separator, prev);
	}
	result.push_back(str.substr(prev));
	return result;
}

std::unordered_map<std::string, std::string>
parseValues(const std::vector<std::string> &received)
{
	std::unordered_map<std::string, std::string> res;
	for (const auto &str : received) {
		auto pos = str.find("=");
		if (pos == std::string::npos) {
			res.insert({ str, "" });
			continue;
		}

		auto key = str.substr(0, pos);
		++pos;
		auto value = str.substr(pos, str.size() - pos);
		res.insert({ key, value });
	}
	return res;
}

std::unordered_map<std::string, std::string>
parseSubValues(const std::string &received)
{
	std::unordered_map<std::string, std::string> res;

	for (const auto &str : splitString(received.substr(1, received.size() - 2),
			";")) {
		auto split = splitString(str, "=");
		if (split.size() == 1) {
			res.insert({ split[0], "" });
		} else {
			res.insert({ split[0], split[1] });
		}
	}

	return res;
}

void PureLiveVideoSocket::stringReceived(const std::string &received)
{
	// Регулярные выражения для распознавания командных строк
	static RegexpMatcher protocolPattern
			("^PROTOCOL ([0-9a-zA-Z ._:-]*)$", RegexpMatcher::EXTENDED);
	static RegexpMatcher getSnapshotPattern
			("^GE(TSMALL|TMIDDLE|T)SNAPSHOT CQ=([0-9]+)$",
					RegexpMatcher::EXTENDED);
	static std::string set_camera_pattern = "SET_CAMERA";
	static std::string get_camera_pattern = "GET_CAMERA";
	static std::string http_get_pattern = "GET";

	auto splited = splitString(received, " ");

	PerfCounter::incrementCounter("lv_strings_received");

	if (!snapshotSendder) {
		snapshotSendder =
				new SnapshotWaiter(this, framequeue, imageformatconverter);
	}

	if (splited[0] == http_get_pattern || snapshotSendder->isHttp()) {
		snapshotSendder->setHttpMode();
		httpGet(splited);
		return;
	}

	if (protocolPattern.match(received)) {
		wasAuthenticated = true;
		std::string clientid = protocolPattern.retrieveMatchedSubstring(1);
		LOGC (DEBUG) << "Client identified: " << clientid << std::endl;
		sendString("PHRP 2.0 Live video connection\n");
		// Заказать следующую строку от клиента
		wantToReadString();
		return;
	}
	if (received == "EXIT") {
		LOGC (DEBUG) << "EXIT command received, closing live video connection"
					<< std::endl;
		closeSocket();
		delete this;
		return;
	}

	// Все последующие команды можно выполнять только после команды
	// PROTOCOL
	if (!wasAuthenticated) {
		LOGC (DEBUG) << "Not authenticated, closing live video connection"
					<< std::endl;
		closeSocket();
		delete this;
		return;
	}
	bool state_ok = true;
	if (getSnapshotPattern.match(received)) {
		PerfCounter::incrementCounter("lv_getsnapshots_received");
		std::string sizename = getSnapshotPattern.retrieveMatchedSubstring(1);
		sizename = sizename.substr(1);
		int cq = atoi(getSnapshotPattern.retrieveMatchedSubstring(2).c_str());
		snapshotSendder->request(cq, sizename);
		return;
	} else if (splited[0] == set_camera_pattern) {
		PerfCounter::incrementCounter("lv_get_frame_received");
		state_ok = setCamera(splited);
	} else if (splited[0] == get_camera_pattern) {
		sendCamera();
	} else {
		state_ok = false;
	}

	if (!state_ok) {
		// Принята неправильная команда
		// Нужно просто закрыть соединение
		LOGC (DEBUG) << "Wrong command, closing live video connection: " << "\""
					<< received << "\"" << std::endl;
		closeSocket();
		delete this;
		return;
	}
	// Заказать чтение следующей строки
	wantToReadString();
}

void PureLiveVideoSocket::httpGet(const std::vector<string> &received)
{
	if (received[0] != "GET") {
		wantToReadString();
		return;
	}

	if (received[1] == "/mjpeg") {
		int w = 1920;
		int h = 1080;
		if (framequeue->getQueueLength() > 0) {
			GrabbedFrameDescriptor *f = framequeue->getFrameByNum(0);
			w = f->width;
			h = f->height;
		}

		// стартуем таск который будет лить mjpeg
		ParamsStr compressparams;
		compressparams.setParam("DESTFORMAT", "JPEG");
		compressparams.setParamInt("DESTQ", 75);
		compressparams.setParamInt("DW", w/2);

		ParamsStr marker; // paint выполняется до resize
		marker.setParam("TYPE", "marker");
		marker.setParamInt("X", w/2);
		marker.setParamInt("Y", h/2);
		marker.setParamInt("R", 20);
		compressparams.setParam("PAINT0", string(";") + marker.toSemicolonString());

		double forced_gamma = 0;
		if (forced_gamma > 0.0) {
			compressparams.setParamDouble("GAMMA", forced_gamma);
		}
		if (!mjpeg_) {
			mjpeg_ = new MjpegSender(this, framequeue, imageformatconverter,
					&compressparams);
			mjpeg_->begin();
		}
	}
}

bool PureLiveVideoSocket::setCamera(const std::vector<string> &received)
{
	float3 rotate;
	auto values = parseValues(received);
	auto it = values.find("ROTATE");
	if (it == values.end()) {
		return false;
	}
	auto args = parseSubValues(it->second);

	it = args.find("X");
	if (it == values.end()) {
		return false;
	}
	rotate.x = std::stof(it->second);

	it = args.find("Y");
	if (it == values.end()) {
		return false;
	}
	rotate.y = std::stof(it->second);

	it = args.find("Z");
	if (it == values.end()) {
		return false;
	}
	rotate.z = std::stof(it->second);

	mesh_reader_->rotateCamEuler(rotate);

	return true;
}

void PureLiveVideoSocket::sendCamera()
{
	float3 rotate = mesh_reader_->getCameraRotation();

	std::stringstream ss;
	ss << "GET_CAMERA";
	ss << " ROTATE=(" << "X=" << rotate.x << ";Y=" << rotate.y << ";Z="
			<< rotate.z << ")";
	ss << "\n";
	sendString(ss.str());
}

void PureLiveVideoSocket::disconnectHappened(int fd)
{
	closeSocket();
	delete this;
}

void PureLiveVideoSocket::errorHappened(int fd)
{
	closeSocket();
	delete this;
}

void PureLiveVideoSocket::endOfFileEncounteredOnInput()
{
	closeSocket();
	delete this;
}

void PureLiveVideoSocket::somethingWasRead()
{
	if (howManyBytesAlreadyRead() > 10000) {
		// Нам пытаются прислать очень длинную строку - видимо, DoS-ят.
		// Надо выругаться и оборвать соединение.
		LOGC (DEBUG) << "Receiving toooooo lengthy string, probably DoS."
					<< " Disconnecting." << std::endl;
		closeSocket();
		delete this;
	}
}

//=========================================================================== 
//========================= PureLiveVideoListeningSocket ========================
//===========================================================================
PureLiveVideoListeningSocket::PureLiveVideoListeningSocket(
		DeferredExecutor *deferredexecutor,
		ImageFormatConverter *imageformatconverter, VideoFrameQueue *framequeue,
		MeshReader *reader):
		deferredexecutor(deferredexecutor),
		imageformatconverter(imageformatconverter), framequeue(framequeue),
		mesh_reader_(reader)
{
}

PureLiveVideoListeningSocket::~PureLiveVideoListeningSocket()
{
	std::set<PureLiveVideoSocket *> connectionscopy = connections;
	for (std::set<PureLiveVideoSocket *>::iterator it = connectionscopy.begin();
			it != connectionscopy.end(); ++it) {
		delete *it;
	}
}

void PureLiveVideoListeningSocket::connectionAccepted(int acceptedsockfd,
		unsigned short int portfrom, u_int32_t ipfrom)
{
	struct in_addr addr;
	addr.s_addr = htonl(ipfrom);
	LOGC (DEBUG) << "New livevideo connection from " << inet_ntoa(addr) << ":"
				<< portfrom << std::endl;

	// Создать новое соединение и подключить его к poller-у
	PureLiveVideoSocket *socket =
			new PureLiveVideoSocket(acceptedsockfd, this, poller,
					deferredexecutor, imageformatconverter, framequeue,
					mesh_reader_);
	connections.insert(socket);
}

PureLiveVideoSocket::MjpegSender::MjpegSender(PureLiveVideoSocket *return_to,
		VideoFrameQueue *framequeue, ImageFormatConverter *imageformatconverter,
		ParamsStr *compressparams):
		return_to(return_to), framequeue(framequeue),
		imageformatconverter(imageformatconverter),
		compressparams(*compressparams)
{
}

PureLiveVideoSocket::MjpegSender::~MjpegSender()
{
	framequeue->delQueueListener(this);
	for (auto id : compression_ids) {
		imageformatconverter->cancelConversion(id);
	}
}

void PureLiveVideoSocket::MjpegSender::begin()
{
	framequeue->addQueueListener(this, target_frame, target_frame);

	// посылаем оновной http ответ
	std::ostringstream s;
	static const std::string STD_HEADER = "Connection: close\r\n"
										  "Server: simicondev/0.0.1\r\n"
										  "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n"
										  "Pragma: no-cache\r\n"
										  "Expires: Mon, 3 Jan 2000 12:34:56 GMT\r\n";

	s << "HTTP/1.0 200 OK\r\n" << STD_HEADER
			<< "Content-Type: multipart/x-mixed-replace;boundary=" << BOUNDARY
			<< "\r\n" << "\r\n" << "--" << BOUNDARY << "\r\n";
	return_to->sendBinaryBlock((unsigned char *) s.str().c_str(),
			s.str().length());
}

void
PureLiveVideoSocket::MjpegSender::frameQueueShifted(VideoFrameQueue *framequeue)
{
	if (target_frame >= framequeue->getQueueLength())
		return;
	if (compression_ids.size() > 2)
		return;
	GrabbedFrameDescriptor *frame = framequeue->getFrameByNum(target_frame);
	ParamsStr cp = compressparams;
	// Оправить на конверсию, когда вернется - запишется в сокет
	if (!cp.hasParam("GAMMA")) { // нет forced_gamma
		// для камеры Макарычева делаем гамму
		if (frame->optimal_gamma > 0.0) {
			cp.setParamDouble("GAMMA", frame->optimal_gamma);
		}
	}
	int compressid = imageformatconverter->startConversion(frame, &cp, this);
	compression_ids.insert(compressid);
}

void PureLiveVideoSocket::MjpegSender::conversionFinished(int taskid,
		FrameDescriptor *result)
{
	compression_ids.erase(taskid);
	if (!result)
		return;

	size_t imagelen = result->getLength();

	PerfCounter::incrementCounter("lv_mjpegframes_sended");
	/*
	 * print the individual mimetype and the length
	 * sending the content-length fixes random stream disruption observed
	 * with firefox
	 */
	std::ostringstream s;
	s << "Content-Type: image/jpeg\r\n" << "Content-Length: " << imagelen
			<< "\r\n" << "\r\n";
	return_to->sendBinaryBlock((unsigned char *) s.str().c_str(),
			s.str().length());

	// Отправить тело сконвертированной картинки
	if (imagelen > 0) {
		BlockWithFrame *blocKwithframe = new BlockWithFrame(result, 0);
		return_to->sendBlock(blocKwithframe);
		blocKwithframe->release();
	}
	std::ostringstream s2;
	s2 << "\r\n--" << BOUNDARY << "\r\n";
	return_to->sendBinaryBlock((unsigned char *) s2.str().c_str(),
			s2.str().length());
}
