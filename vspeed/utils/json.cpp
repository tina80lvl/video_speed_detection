#include <fstream>
#include <sstream>
#include <iostream>
#include "json.h"

rapidjson::Document JsonParser::getJsonDocument(const std::string &filename) {
    // open the JSON file
    std::ifstream jFile(filename);
    if (!jFile.is_open())
        throw JsonParserException("Unable to open file " + filename + ".");


//     read the file contents
    std::stringstream contents;
    contents << jFile.rdbuf();

    rapidjson::Document doc;
    // parse the JSON from the string contents
    doc.Parse(contents.str().c_str());

    if (doc.HasParseError()) {
        std::cout << "Error  : " << doc.GetParseError() << '\n'
                  << "Offset : " << doc.GetErrorOffset() << '\n'
                  << "Text   : " << GetParseError_En(doc.GetParseError())
                  << '\n';
        return nullptr;
    }
    return doc;
}

std::string JsonParser::getJsonText(const rapidjson::Document &doc) {
    rapidjson::StringBuffer buffer;

    buffer.Clear();

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    return {buffer.GetString()};
}
