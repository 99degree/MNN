#!/bin/bash
# Build MNN with full profiling support
# Usage: ./build_profiling.sh [clean|all]

set -e

BUILD_DIR="build_profile"
INSTALL_DIR="install_profile"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clean build
if [ "$1" = "clean" ]; then
    rm -rf "$BUILD_DIR" "$INSTALL_DIR"
    echo "Cleaned build directory"
    exit 0
fi

# CMake configuration with profiling enabled
echo "Configuring MNN with profiling support..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_GPU_TIME_PROFILE=ON \
    -DMNN_BUILD_TEST=ON \
    -DMNN_INTERNAL=ON \
    -DMNN_ENABLE_COVERAGE=OFF \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_BUILD_TRAIN=OFF \
    -DMNN_BUILD_CONVERTER=ON \
    -DMNN_BUILD_QUANTOOLS=OFF \
    -DCMAKE_INSTALL_PREFIX=../"$INSTALL_DIR" \
    -DANDROID_ABI="" \
    2>&1 | tee ../build_config.log

echo "Building MNN with profiling..."
make -j$(nproc) 2>&1 | tee ../build_log.txt

echo "Installing MNN..."
make install 2>&1 | tee ../install_log.txt

echo ""
echo "✅ MNN built with profiling support!"
echo ""
echo "Build artifacts:"
echo "  - Libraries: $INSTALL_DIR/lib/"
echo "  - Headers: $INSTALL_DIR/include/"
echo "  - Build directory: $BUILD_DIR/"
echo ""
echo "Available profiling APIs:"
echo "  - Interpreter::getSessionInfo(session, MEMORY, &float_value)"
echo "  - Interpreter::getSessionInfo(session, FLOPS, &float_value)"
echo "  - Interpreter::getSessionInfo(session, BACKENDS, int_array)"
echo "  - Interpreter::getSessionInfo(session, RESIZE_STATUS, &int_value)"
echo "  - Interpreter::getSessionInfo(session, THREAD_NUMBER, &int_value)"
echo ""
echo "GPU profiling enabled: OpenCL and Vulkan time profiling"
cd ..