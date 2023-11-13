cmake_minimum_required(VERSION 3.18)

set(DLIB_PATH /movie/dlib/dlib)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

cuda_add_executable(sample sample.cpp)

target_include_directories(sample PUBLIC "/usr/local/include/opencv4")
target_include_directories(sample PUBLIC "/")

target_link_libraries(sample opencv_core opencv_highgui opencv_dnn tracker)