cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project( gender_classifier )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( src )

#set(HEADER_FILES )
set(SOURCES src/main.cpp)

add_executable( gender_classifier ${SOURCES}) #${HEAD_FILES})

target_link_libraries( gender_classifier ${OpenCV_LIBS} )