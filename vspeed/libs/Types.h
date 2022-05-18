#ifndef _1_TYPES_H_
#define _1_TYPES_H_

#include <string>

#include "OpenCVFwd.h"

using std::string;

#define     INT16        short
#define     UINT16        unsigned short
#define     FLOAT64            double

#define     TRUE            1
#define     FALSE           0

#define     MAXCERT                (100.0)

#define     MAX_SYMBOL_NUM            (16)

typedef INT16 TInt3[4];

enum class LicnumEqualityType {
    BY_FORMAT,  // сравниваем только формат
    BY_SYMBOLS, // полное соответствие символов
    BY_FORMAT_AND_SYMBOLS // полное соответствие символов и формата
};

struct Rectangle;
typedef struct TRectNumber {
    int numFormat = 0;
    int n_symbols = 0;
    UINT16 text16[MAX_SYMBOL_NUM] = {};

    INT16 allCert = {};
    INT16 certList[MAX_SYMBOL_NUM] = {};
    TInt3 x = {};
    TInt3 y = {};
    /* Нумерация вершин:
     * 1--------------3
     * | A 999 AA 178 |
     * 2--------------0
     */
public:
    int centerX() const;

    int centerY() const;

    cv::Point center() const;

    int height() const;

    int width() const;

    void setUpperLeft(short new_x, short new_y);

    void setUpperRight(short new_x, short new_y);

    void setLowerLeft(short new_x, short new_y);

    void setLowerRight(short new_x, short new_y);

    bool operator==(const TRectNumber &other) const;

    bool isLicnumEqual(const TRectNumber &other,
                       LicnumEqualityType equality_type) const;

    bool isLicnumEqualByFormat(const TRectNumber &other) const;

    bool isLicnumEqualByFormatAndSymbols(const TRectNumber &other) const;

    bool isLicnumEqualBySymbols(const TRectNumber &other) const;

    bool isSizeValid() const;

    // Проверка номер однострочный или двухстрочный
    bool isDoubleLine() const;

    std::string toString() const;

    Rectangle boundingRectangle() const;
} TRectNumber;

std::ostream &operator<<(std::ostream &stream, const TRectNumber &licnum);

typedef struct TRectNumberCp1251 {
    TInt3 x, y;
    char number[12];
    INT16 allCert;
    int numFormat;
    INT16 certList[12];
    UINT16 Time;
} TRectNumberCp1251;

enum LikeAlgorithm {
    DIRECT_LIKE_ALGORITHM,        // сравниваем строки номеров без сдвига
    SHIFTING_LIKE_ALGORITHM       // сравниваем строки номеров со сдвигом
};

enum class TcpXmlAckMode {
    SKIPPED = 0,
    ACCEPTED = 1,
    UNKNOWN = 2,
    REJECTED_BY_BAD_JPEG = 3,
    REJECTED_DUE_TO_ERROR = 4
};

enum class RadarModel : uint32_t {
    UMRR0A = 0,
    UMRR11 = 1
};

string to_string(TcpXmlAckMode mode);

#endif

#ifndef WHAT_TO_DO_FOR_SIGN_HELPER
#define WHAT_TO_DO_FOR_SIGN_HELPER
enum WhatToDoForDataSignHelper {
    NOTHING_TO_DO,
    HASH,
    VERIFY_WITH_HASH,
    VERIFY_WITHOUT_HASH,
    SIGN_WITH_HASH,
    SIGN_WITHOUT_HASH
};
#endif
