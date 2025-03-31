cmake_minimum_required(VERSION 3.18) 
enable_language(CUDA)  # 关键行
set(CMAKE_CUDA_ARCHITECTURES "86") 
set(CMAKE_CUDA_STANDARD 17)         # 使用C++17标准
set(CMAKE_CXX_STANDARD 17)

file(GLOB INFER_SOURCES "src/*.cpp" "src/op/*.cpp" "src/op/cpu_kernel/*.cpp" "src/op/gpu_kernel/*.cpp") 
file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/op/gpu_kernel/*.cu")
message(STATUS "CUDA source files: ${CUDA_SOURCES}")
project(infer LANGUAGES CXX CUDA) 
# 添加库
add_library(infer STATIC
    ${INFER_SOURCES}
    ${CUDA_SOURCES}
)

# 针对CUDA文件设置属性
set_target_properties(infer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared  
)

# 包含头文件目录
target_include_directories(infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op/cpu_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op/gpu_kernel
    ${CUDA_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(infer PUBLIC armadillo glog::glog ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cudart_LIBRARY} )