#ifndef LOG_H_INCLUDED
#define LOG_H_INCLUDED

#include <iostream>

using namespace std;

enum LogFlag {
    LOGC_DEBUG,
    LOGC_ERROR,
    LOGC_WARNING,
    LOGC_INFO,
    LOGC_OUTPUT_LOG_LEVEL,
    LOGC_OUTPUT_FUNCTION_NAME,
    LOGC_OUTPUT_FILE_AND_LINE,
    LOGC_OUTPUT_NEW_LINE_BEFORE_MESSAGE,

    LOGC_IPDLSOCK = 14, /* ImageProxyDownlinkSocket::printSendingBufferDebugInfo */
    LOGC_POLLER = 15, /* FileDescriptorPoller DEBUG messages */

    //placeholder
    LOGC_APP1 = 16,

    //application-specific
    LOGC_SENSORAPP_SEQUENCE = LOGC_APP1, /* comment my usage */

    //serialserver specific now, but can be used by others
    LOGC_CAMERA = 17,
    LOGC_RADAR = 18,
    LOGC_MVIDEO = 19,
    LOGC_ULTRASOUND = 20,
    LOGC_US = 21
};

//Logging interface
bool log_init_logger(const char *appname, int instance);

bool log_init_logger2(const char *argv0);

void logc_printf(LogFlag level, const char *format, ...);

#define LOGIF(level) if (!is_log_flag_contained (LOGC_ ## level)); else
#define LOGC(level) \
    if (!is_log_flag_contained(LOGC_ ## level)); \
    else write_formatted_log_header(#level, __PRETTY_FUNCTION__, __FILE__, __LINE__)
#define LOGP(level) (std::cout)

//Control interface
bool log_init_control();

#define LOG_MAX_INSTANCES (16)
#define LOG_MAX_APPLICATIONS (128)

//application IDs
struct log_applications_id_t {
    const char *execname;
    int id;
};

constexpr log_applications_id_t log_applications_id[] = {
        {"other",               0},
        {"sensorapp",           1},
        {"signserver",          2},
        {"videorecorder",       3},
        {"volmngr",             4},
        {"glonass",             5},
        {"imageproxy",          6},
        {"broadcastresponder",  7},
        {"jpegserver",          8},
        {"storagemanager",      9},
        {"statserver",          10},
        {"dbserver",            11},
        {"recog",               12},
        {"distributedstorage",  13},
        {"telemetry",           14},
        {"committee",           15},
        {"modem",               16},
        {"violation",           17},
        {"crossroad",           18},
        {"crosssensordetector", 19}
// tool are all in 'other' category
};

constexpr unsigned LOG_APPLICATIONMAP_ID_COUNT =
        sizeof(log_applications_id) / sizeof(*log_applications_id);

//for usage only throught macroses
extern volatile unsigned *log_mask; //lay on shm

bool is_log_flag_contained(LogFlag log_level);

ostream &
write_formatted_log_header(const char *log_level, const char *function_name,
                           const char *file_name, int line_number,
                           ostream &stream = cout);

//real value of instances
struct log_shm_table_t {
    unsigned logl[LOG_MAX_INSTANCES * LOG_MAX_APPLICATIONS];
};

extern log_shm_table_t *log_shm_table;

#endif //LOG_H_INCLUDED
