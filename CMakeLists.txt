# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.14)

# 设置项目名称和版本
project(SyInfer VERSION 1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB INFER_SOURCES "infer/src/*.cpp")

#外部依赖
find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)

# 添加可执行文件
add_executable(SyInfer
    ${INFER_SOURCES}
    infer/infer.cpp
)

# 包含头文件目录
target_include_directories(SyInfer PUBLIC
    ${CMAKE_SOURCE_DIR}/infer/include
)

# 添加库（如果有）
target_link_libraries(SyInfer PRIVATE armadillo gtest)

# 安装目标（可选）

add_subdirectory(infer)

# 测试（可选）
enable_testing()
add_subdirectory(tests)