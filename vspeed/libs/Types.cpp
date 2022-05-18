#include "Types.h"
//#include "ICONVConverter.h"

#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <codecvt>
#include "Rectangle.h"

using namespace std;

int TRectNumber::centerX() const {
    return (x[0] + x[1] + x[2] + x[3]) / 4;
}

int TRectNumber::centerY() const {
    return (y[0] + y[1] + y[2] + y[3]) / 4;
}

cv::Point TRectNumber::center() const {
    return {centerX(), centerY()};
}

void TRectNumber::setUpperLeft(short new_x, short new_y) {
    x[1] = new_x;
    y[1] = new_y;
}

void TRectNumber::setUpperRight(short new_x, short new_y) {
    x[3] = new_x;
    y[3] = new_y;
}

void TRectNumber::setLowerLeft(short new_x, short new_y) {
    x[2] = new_x;
    y[2] = new_y;
}

void TRectNumber::setLowerRight(short new_x, short new_y) {
    x[0] = new_x;
    y[0] = new_y;
}

int TRectNumber::height() const {
    // С учетом параллелограммости номера, приходящего с распознавания,
    // высоту можно посчитать по одной стороне
    return y[2] - y[1];
}

int TRectNumber::width() const {
    return x[3] - x[1];
}

bool TRectNumber::operator==(const TRectNumber &other) const {
    return numFormat == other.numFormat && n_symbols == other.n_symbols
           && allCert == other.allCert && equal(x, std::end(x), other.x)
           && equal(y, std::end(y), other.y)
           && equal(text16, text16 + n_symbols, other.text16)
           && equal(certList, certList + n_symbols, other.certList);
}

bool TRectNumber::isLicnumEqual(const TRectNumber &other,
                                LicnumEqualityType equality_type) const {
    switch (equality_type) {
        case LicnumEqualityType::BY_FORMAT:
            return numFormat == other.numFormat;
        case LicnumEqualityType::BY_SYMBOLS:
            if (n_symbols != other.n_symbols) {
                return false;
            }
            return equal(text16, text16 + n_symbols, other.text16);
        case LicnumEqualityType::BY_FORMAT_AND_SYMBOLS:
            return isLicnumEqual(other, LicnumEqualityType::BY_FORMAT)
                   && isLicnumEqual(other, LicnumEqualityType::BY_SYMBOLS);
        default:
            return false;
    }
}

bool TRectNumber::isLicnumEqualByFormat(const TRectNumber &other) const {
    return isLicnumEqual(other, LicnumEqualityType::BY_FORMAT);
}

bool
TRectNumber::isLicnumEqualByFormatAndSymbols(const TRectNumber &other) const {
    return isLicnumEqual(other, LicnumEqualityType::BY_FORMAT_AND_SYMBOLS);
}

bool TRectNumber::isLicnumEqualBySymbols(const TRectNumber &other) const {
    return isLicnumEqual(other, LicnumEqualityType::BY_SYMBOLS);
}

bool TRectNumber::isSizeValid() const {
    return width() && height();
}

// Проверка номер однострочные или двухстрочный
bool TRectNumber::isDoubleLine() const {
    const double double_line_aspect_ration = 0.5;

    // Отношение высоты к ширине номера. Если оно больше, чем
    // double_line_aspect_ration, считаем, что номер двухстрочный
    const double aspect_ratio = static_cast<double>(height()) / width();

    return aspect_ratio > double_line_aspect_ration;
}

string TRectNumber::toString() const {
    std::u16string s(text16, text16 + n_symbols);
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> convert;
    return convert.to_bytes(s);
//	return ICONVConverter::convertUtf16toUtf8(text16, n_symbols);
//	return "text16";
}

Rectangle TRectNumber::boundingRectangle() const {
    const auto[min_x_it, max_x_it] = minmax_element(cbegin(x), cend(x));
    const auto[min_y_it, max_y_it] = minmax_element(cbegin(y), cend(y));
    const auto w = *max_x_it - *min_x_it;
    const auto h = *max_y_it - *min_y_it;

    return Rectangle(*min_x_it, *min_y_it, w, h);
}

ostream &operator<<(ostream &stream, const TRectNumber &licnum) {
    stream << "TRectNumber('" << licnum.toString() << "', x,y: (";
    auto print_x_y = [&](size_t i) {
        stream << '{' << licnum.x[i] << ", " << licnum.y[i] << '}';
    };

    print_x_y(0);
    for (size_t i = 1; i < 4; ++i) {
        stream << ", ";
        print_x_y(i);
    }
    return stream << "))";
}

namespace {
    constexpr string_view ack_mode_skipped{"skipped"};
    constexpr string_view ack_mode_accepted{"accepted"};
    constexpr string_view ack_mode_rejected_by_bad_jpeg{"rejected_by_bad_jpeg"};
    constexpr string_view ack_mode_rejected_due_to_error{
            "rejected_due_to_error"};
    constexpr string_view ack_mode_unknown{"unknown"};
}

string to_string(TcpXmlAckMode mode) {
    string_view mode_str;
    switch (mode) {
        case TcpXmlAckMode::SKIPPED:
            mode_str = ack_mode_skipped;
            break;
        case TcpXmlAckMode::ACCEPTED:
            mode_str = ack_mode_accepted;
            break;
        case TcpXmlAckMode::REJECTED_BY_BAD_JPEG:
            mode_str = ack_mode_rejected_by_bad_jpeg;
            break;
        case TcpXmlAckMode::REJECTED_DUE_TO_ERROR:
            mode_str = ack_mode_rejected_due_to_error;
            break;
        case TcpXmlAckMode::UNKNOWN:
            [[fallthrough]];
        default:
            mode_str = ack_mode_unknown;
    }
    return string(mode_str);
}

