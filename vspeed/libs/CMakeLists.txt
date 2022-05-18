project(libs)

#find_package(OpenCV REQUIRED)

set(SOURCES Calc.cpp Log.cpp Rectangle.cpp Transform.cpp Types.cpp)
set(HEADERS Calc.h Log.h Rectangle.h Transform.h Types.h OpenCVFwd.h)

add_library(lib ${SOURCES} ${HEADERS})

target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

