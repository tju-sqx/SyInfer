# ... existing code ...
# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.14)

# 设置项目名称和版本
project(SyInfer VERSION 1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#设置cuda标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86") # 根据您的GPU型号调整

# 外部依赖
find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(CUDA REQUIRED)

# 添加infer子目录
add_subdirectory(infer)

# 添加可执行文件
add_executable(SyInfer infer/infer.cpp)

# 包含头文件目录
target_include_directories(SyInfer PUBLIC
    ${CMAKE_SOURCE_DIR}/infer/include
)

# 链接库
target_link_libraries(SyInfer PRIVATE infer armadillo gtest glog::glog ${CUDA_LIBRARIES})

# 测试（可选）
enable_testing()
add_subdirectory(tests)