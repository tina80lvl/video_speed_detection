project(vspeed-utils)

set(SOURCES utils.cpp json.cpp)
set(HEADERS utils.h json.h)

add_library(utils ${SOURCES} ${HEADERS})
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(RapidJSON CONFIG REQUIRED)
target_link_libraries(utils PUBLIC rapidjson)