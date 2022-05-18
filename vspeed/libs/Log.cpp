#include "Log.h"

#ifndef __MINGW32__

#include <sys/mman.h>

#endif

#include <cerrno>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <cstdlib>

#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <cstdarg>     /* va_list, va_start, va_arg, va_end */
#include <iomanip>

#define FILENAME ("/tmp/phradar_logsettings")

static unsigned log_mask_zero;

volatile unsigned *log_mask = &log_mask_zero; //Turn off logs by default
#ifdef __MINGW32__
unsigned log_mask_win;
#endif

log_shm_table_t *log_shm_table;

#ifndef __MINGW32__

bool log_shm_opencreate(int mode) {
    int fd = open(FILENAME, O_RDWR | O_CREAT, 00666);
    if (fd == -1)
        return false;

    struct stat buf;
    if (fstat(fd, &buf) == -1)
        return false;

    if (buf.st_size <
        (int) sizeof(log_shm_table_t)) { //newly created thing or just older version
        if (ftruncate(fd, sizeof(log_shm_table_t)) == -1)
            return false;

        //set default
        void *map = mmap(0, sizeof(log_shm_table_t), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
        if (map == MAP_FAILED)
            return false;
        log_shm_table_t *log_shm_table_def = (log_shm_table_t *) map;
        for (int i = 0; i < LOG_MAX_INSTANCES * LOG_MAX_APPLICATIONS; i++) {
            log_shm_table_def->logl[i] = (1 << LOGC_INFO) |
                                         (1 << LOGC_WARNING) |
                                         (1 << LOGC_ERROR);
        }
        munmap(map, sizeof(log_shm_table_t));
    }

    if (mode == O_RDONLY) { //now we have shm of correct size
        close(fd);
        fd = open(FILENAME, O_RDONLY, 00666);
    }

    int prot = PROT_READ;
    if ((mode & O_RDWR) == O_RDWR)
        prot = PROT_READ | PROT_WRITE;

    void *map = mmap(0, sizeof(log_shm_table_t), prot, MAP_SHARED, fd, 0);

    if (map == MAP_FAILED)
        return false;

    log_shm_table = (log_shm_table_t *) map;

    return true;
}

#endif //__MINGW32__

bool log_init_logger(const char *appname, int instance) {
    if (instance < 0)
        instance = 0;
    if (instance >= LOG_MAX_INSTANCES)
        instance = LOG_MAX_INSTANCES - 1;

#ifndef __MINGW32__
    if (!log_shm_opencreate(O_RDONLY))
        return false;
#endif //__MINGW32__

    int appid;
    bool found = false;
    unsigned int id;

    for (id = 0; id < LOG_APPLICATIONMAP_ID_COUNT; id++) {
        if (strcmp(log_applications_id[id].execname, appname) == 0
            || "rt-" + std::string(log_applications_id[id].execname)
               == appname) {
            found = true;
            break;
        }
    }

    if (found) {
        appid = log_applications_id[id].id;
    } else
        appid = 0;

#ifndef __MINGW32__
    log_mask = &(log_shm_table->logl[LOG_MAX_INSTANCES * appid + instance]);
#else  //__MINGW32__
    log_mask=&log_mask_win;
    *log_mask=(1<<LOG_DEBUG)|(1<<LOG_INFO)|(1<<LOG_WARNING)|(1<<LOG_ERROR);
#endif //__MINGW32__

    return true;
}

bool log_init_logger2(const char *argv0) {
    const char *pos = strrchr(argv0, '/');
    const char *name;
    if (pos) {
        name = pos + 1;
    } else
        name = argv0;
    int instance = 0;
    char *val = getenv("INSTANCENUM");
    if (val) {
        instance = atoi(val);
    }
    return log_init_logger(name, instance);
}

void logc_printf(LogFlag level, const char *format, ...) {
    if (!(*log_mask & (1 << level))) {
        return;
    }

    va_list argptr;
    va_start(argptr, format);
    vprintf(format, argptr);
    va_end(argptr);

    fflush(stdout);
}

bool log_init_control() {
#ifndef __MINGW32__
    return log_shm_opencreate(O_RDWR);
#else //__MINGW32__
    return false;
#endif
}

namespace {
    bool has_log_mask_any_of_bits(unsigned log_bits) {
        return *log_mask & log_bits;
    }
}

bool is_log_flag_contained(LogFlag log_level) {
    const unsigned log_level_mask = 1 << log_level;
    return has_log_mask_any_of_bits(log_level_mask);
}

ostream &
write_formatted_log_header(const char *log_level, const char *function_name,
                           const char *file_name, int line_number,
                           ostream &stream) {

    constexpr unsigned output_flag_bits =
            ((1 << LOGC_OUTPUT_LOG_LEVEL) | (1 << LOGC_OUTPUT_FUNCTION_NAME)
             | (1 << LOGC_OUTPUT_FILE_AND_LINE)
             | (1 << LOGC_OUTPUT_NEW_LINE_BEFORE_MESSAGE));

    if (!has_log_mask_any_of_bits(output_flag_bits)) {
        // Данная проверка введена с целью ускорить обычное поведение лога
        // ERROR WARNING INFO (DEBUG)
        return stream;
    }

    if (is_log_flag_contained(LOGC_OUTPUT_LOG_LEVEL)) {
        // необходимо для выравнивания. WARNING - 7 символов
        constexpr int maximum_error_level_string_length = 7;
        stream << '[' << setw(maximum_error_level_string_length) << log_level
               << "] ";
    }

    if (is_log_flag_contained(LOGC_OUTPUT_FUNCTION_NAME)) {
        stream << function_name << ' ';
    }

    if (is_log_flag_contained(LOGC_OUTPUT_FILE_AND_LINE)) {
        stream << file_name << ':' << line_number << ' ';
    }

    if (is_log_flag_contained(LOGC_OUTPUT_NEW_LINE_BEFORE_MESSAGE)) {
        stream << '\n';
    }

    return stream;
}
