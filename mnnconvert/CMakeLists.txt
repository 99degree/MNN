cmake_minimum_required(VERSION 3.6)
project(mnnconvert VERSION 1.0.0 LANGUAGES C CXX ASM)

# ============================================================
# Dependencies: MNN core, protobuf, flatbuffers, rapidjson
# ============================================================

# Find MNN core library
if(NOT MNN_LIBRARY)
    find_library(MNN_LIBRARY NAMES MNN libMNN PATHS ${MNN_DIR}/lib ${MNN_DIR}/lib64 NO_DEFAULT_PATH)
    if(NOT MNN_LIBRARY)
        find_library(MNN_LIBRARY NAMES MNN libMNN)
    endif()
endif()
if(NOT MNN_INCLUDE_DIR)
    find_path(MNN_INCLUDE_DIR MNN/MNNDefine.h PATHS ${MNN_DIR}/include ${CMAKE_CURRENT_LIST_DIR}/../include NO_DEFAULT_PATH)
    if(NOT MNN_INCLUDE_DIR)
        find_path(MNN_INCLUDE_DIR MNN/MNNDefine.h)
    endif()
endif()

message(STATUS "MNN library: ${MNN_LIBRARY}")
message(STATUS "MNN include: ${MNN_INCLUDE_DIR}")

# Dependencies from MNN's 3rd_party
set(MNN_3RD_PARTY ${CMAKE_CURRENT_LIST_DIR}/../3rd_party)

# FlatBuffers
set(FLATBUFFERS_SRC ${MNN_3RD_PARTY}/flatbuffers/src/util.cpp)

# Protobuf
option(MNN_BUILD_PROTOBUFFER "Use built-in protobuf" ON)
if(MNN_BUILD_PROTOBUFFER)
    set(Protobuf_LIBRARIES libprotobuf)
    include_directories(${MNN_3RD_PARTY}/protobuf/src)
else()
    find_package(Protobuf REQUIRED)
    if(${CMAKE_VERSION} VERSION_LESS "3.6.0")
        set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
        set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
    endif()
endif()

# Converter source root (points to original tools/converter/, files remain unchanged)
set(CONVERTER_DIR ${CMAKE_CURRENT_LIST_DIR}/../tools/converter)

# ============================================================
# ONNX converter (unchanged, from original location)
# ============================================================
file(GLOB ONNX_SRC ${CONVERTER_DIR}/source/onnx/*.cpp)
file(GLOB ONNX_PROTO_SRC ${CONVERTER_DIR}/source/onnx/generated/*.cc)

# ============================================================
# Common sources (unchanged)
# ============================================================
file(GLOB COMMON_SRC ${CONVERTER_DIR}/source/common/*.cpp)

# ============================================================
# MNN sources (unchanged) - addBizCode etc.
# ============================================================
file(GLOB MNN_SRC ${CONVERTER_DIR}/source/MNN/*.cpp)

# ============================================================
# Optimizer - exclude non-ONNX backends
# ============================================================
file(GLOB_RECURSE OPTIMIZER_SRC ${CONVERTER_DIR}/source/optimizer/*.cpp)
list(FILTER OPTIMIZER_SRC EXCLUDE REGEX ".*/caffeextra/.*")
list(FILTER OPTIMIZER_SRC EXCLUDE REGEX ".*/tfextra/.*")
list(FILTER OPTIMIZER_SRC EXCLUDE REGEX ".*/tflitextra/.*")
list(FILTER OPTIMIZER_SRC EXCLUDE REGEX ".*/torchextra/.*")

# ============================================================
# Compression (unchanged)
# ============================================================
file(GLOB COMPRESSION_SRC ${CONVERTER_DIR}/source/compression/*.cpp)
file(GLOB COMPRESSION_PROTO_SRC ${CONVERTER_DIR}/source/compression/generated/*.cc)

# ============================================================
# All converter sources (shared between exe and shared lib)
# ============================================================
set(CONVERTER_SOURCES
    ${ONNX_SRC}
    ${ONNX_PROTO_SRC}
    ${COMMON_SRC}
    ${MNN_SRC}
    ${OPTIMIZER_SRC}
    ${COMPRESSION_SRC}
    ${COMPRESSION_PROTO_SRC}
    ${FLATBUFFERS_SRC}
)

# Find protobuf include dir
find_path(PROTOBUF_INCLUDE_DIR google/protobuf/stubs/common.h PATHS ${MNN_3RD_PARTY}/protobuf/src NO_DEFAULT_PATH)
if(NOT PROTOBUF_INCLUDE_DIR)
    find_path(PROTOBUF_INCLUDE_DIR google/protobuf/stubs/common.h)
endif()
message(STATUS "Protobuf include dir: ${PROTOBUF_INCLUDE_DIR}")

# Common include directories
set(MNN_CONVERT_INCLUDES
    ${CONVERTER_DIR}/source/onnx/generated
    ${CONVERTER_DIR}/source/compression/generated
    ${CONVERTER_DIR}/source/tflite/schema
    ${CONVERTER_DIR}/include
    ${CONVERTER_DIR}/source/optimizer
    ${MNN_INCLUDE_DIR}
    ${MNN_3RD_PARTY}/flatbuffers/include
    ${MNN_3RD_PARTY}/rapidjson/include
    ${MNN_3RD_PARTY}/half
    ${CMAKE_CURRENT_LIST_DIR}/../schema/default
    ${CMAKE_CURRENT_LIST_DIR}/../schema/current
    ${CMAKE_CURRENT_LIST_DIR}/../source
    ${CMAKE_CURRENT_LIST_DIR}/../source/core
    ${CMAKE_CURRENT_LIST_DIR}/../source/geometry
    ${CMAKE_CURRENT_LIST_DIR}/../source/backend
    ${CMAKE_CURRENT_LIST_DIR}/../source/shape
    ${PROTOBUF_INCLUDE_DIR}
)

# Common compile definitions
set(MNN_CONVERT_DEFS GOOGLE_PROTOBUF_NO_RTTI)

# ============================================================
# Build executable (CLI tool)
# ============================================================
add_executable(mnnconvert
    ${CMAKE_CURRENT_LIST_DIR}/source/mnnconvert_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/source/cli_stripped.cpp
    ${CONVERTER_SOURCES}
)

set_property(TARGET mnnconvert PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(mnnconvert PRIVATE ${MNN_CONVERT_INCLUDES})
target_compile_definitions(mnnconvert PRIVATE ${MNN_CONVERT_DEFS})
target_link_libraries(mnnconvert
    ${MNN_LIBRARY}
    ${Protobuf_LIBRARIES}
    pthread
)

# ============================================================
# Build shared library (optional, for SDK use)
# ============================================================
option(MNN_BUILD_CONVERTER_SHARED "Build mnnconvert as a shared library" OFF)

if(MNN_BUILD_CONVERTER_SHARED)
    add_library(mnnconvert_shared SHARED
        ${CONVERTER_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/source/mnnconvert_shared.cpp
        ${CMAKE_CURRENT_LIST_DIR}/source/cli_stripped.cpp
    )

    set_property(TARGET mnnconvert_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(mnnconvert_shared PRIVATE ${MNN_CONVERT_INCLUDES})
    target_include_directories(mnnconvert_shared PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
    )
    target_compile_definitions(mnnconvert_shared PRIVATE ${MNN_CONVERT_DEFS} MNN_CONVERTER_SHARED)
    target_link_libraries(mnnconvert_shared
        ${MNN_LIBRARY}
        ${Protobuf_LIBRARIES}
        pthread
    )

    if(UNIX AND NOT APPLE)
        set_target_properties(mnnconvert_shared PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
        )
    endif()
endif()