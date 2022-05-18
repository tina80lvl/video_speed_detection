project(vspeed-algo)

set(SOURCES CalcSpeed.cpp MapFrames.cpp)
set(HEADERS CalcSpeed.h MapFrames.h)

add_library(algo ${SOURCES} ${HEADERS})
target_include_directories(algo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(RapidJSON CONFIG REQUIRED)
target_link_libraries(algo PUBLIC rapidjson)