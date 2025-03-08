file(GLOB INFER_SOURCES "src/*.cpp" "src/op/*.cpp")
# 添加库
add_library(infer STATIC
    ${INFER_SOURCES}
)

# 包含头文件目录
target_include_directories(infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/op
)

# 链接库
target_link_libraries(infer PUBLIC armadillo glog::glog)