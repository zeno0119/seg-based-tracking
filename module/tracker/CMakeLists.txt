cmake_minimum_required(VERSION 3.18)
project(tracker)
add_subdirectory(${DLIB_PATH} dlib_build)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} STATIC ${SRCS})
target_compile_options(${PROJECT_NAME} PUBLIC "-Wall")
target_link_libraries(${PROJECT_NAME} opencv_core dlib::dlib opencv_highgui inference_engine)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC "/usr/local/include/opencv4")
