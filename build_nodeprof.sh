#!/bin/bash
# Build MNN with full per‑node profiling (CPU + Express)
# Usage: ./build_nodeprof.sh

set -e

BUILD_DIR="build_nodeprof"
INSTALL_DIR="install_nodeprof"

# Clean previous build if requested
if [ "$1" == "clean" ]; then
    rm -rf "$BUILD_DIR" "$INSTALL_DIR"
    echo "Cleaned build directories"
    exit 0
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with profiling flags
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_PIPELINE_PROFILE=ON \
    -DMNN_EXPR_ENABLE_PROFILER=ON \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_BUILD_TEST=ON \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_INTERNAL=OFF

# Build
make -j$(nproc)

# Install (optional)
make install DESTDIR=../"$INSTALL_DIR"

echo "✅ Build completed with per‑node profiling flags"
