cmake_minimum_required(VERSION 3.10)
project(MNNProfilingTest)

# Find or configure MNN
set(MNN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Include MNN source
add_subdirectory(${MNN_SOURCE_DIR})

# Create test executable
add_executable(test_profiling_api test_profiling_api.cpp)

# Link against MNN
target_link_libraries(test_profiling_api MNN)

# Include directories
target_include_directories(test_profiling_api PRIVATE 
    ${MNN_SOURCE_DIR}/include
)

# Enable C++11
target_compile_features(test_profiling_api PRIVATE cxx_std_11)

# Output directory
set_target_properties(test_profiling_api PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)