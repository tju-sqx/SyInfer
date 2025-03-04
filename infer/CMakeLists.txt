add_library(infer
    src/alloc.cpp
    src/alloc_cpu.cpp
    src/base.cpp
    src/buffer.cpp
    src/rmsnorm_kernel_cpu.cpp
    src/tensor.cpp
)

# 包含头文件目录
target_include_directories(infer PUBLIC include)

# 链接Armadillo库
target_link_libraries(infer PUBLIC armadillo)