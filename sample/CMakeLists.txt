cmake_minimum_required(VERSION 3.18)

set(DLIB_PATH /movie/dlib/dlib)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

cuda_add_executable(sample sample.cpp)
add_executable(vis visualize.cpp)

target_compile_options(vis PUBLIC "-Wall")

target_link_libraries(sample opencv_core opencv_highgui opencv_dnn tracker)
target_link_libraries(vis opencv_core opencv_highgui opencv_imgproc)