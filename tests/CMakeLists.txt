add_executable(test_rmsnorm test_rmsnorm.cpp)
target_link_libraries(test_rmsnorm PRIVATE GTest::GTest GTest::Main infer)

add_test(NAME test_rmsnorm COMMAND test_rmsnorm)