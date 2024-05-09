cmake_minimum_required(VERSION 3.27)

set(CMAKE_OSX_ARCHITECTURES arm64)
set(APP_NAME mnist)

project(${APP_NAME} CXX)

set(CMAKE_CXX_STANDARD 17)

include(CPM.cmake)

CPMAddPackage(
    VERSION 0.12.2
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
)

CPMAddPackage(
    GIT_TAG 5.0
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
)

include_directories(${mlx_SOURCE_DIR})

include_directories(${raylib_SOURCE_DIR}/src)

add_executable(${APP_NAME} ${APP_NAME}.cpp)
add_executable(${APP_NAME}-viz ${APP_NAME}-viz.cpp)

target_link_libraries(${APP_NAME} PRIVATE mlx)
target_link_libraries(${APP_NAME}-viz PRIVATE mlx raylib)
