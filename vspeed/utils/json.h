#ifndef VSPEED_JSON_H
#define VSPEED_JSON_H

#include <string>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include "rapidjson/error/en.h"


class JsonParserException : public std::runtime_error {
public:
    ~JsonParserException() {}

    JsonParserException(const std::string msg)
            : std::runtime_error(msg) {}
};

class JsonParser {
public:
    static rapidjson::Document getJsonDocument(const std::string &filename);

    static std::string getJsonText(const rapidjson::Document &doc);
};


#endif //VSPEED_JSON_H
