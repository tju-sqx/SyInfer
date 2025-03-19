add_executable(test_op test_rmsnorm.cpp test_swiglu.cpp test_softmax.cpp test_emb.cpp test_rope.cpp)
target_link_libraries(test_op PRIVATE GTest::GTest GTest::Main infer)

add_test(NAME test_op COMMAND test_op)