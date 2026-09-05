#!/bin/bash
set -e

BUILD_DIR="${1:-build_isp}"

cd "$(dirname "$0")"

echo "Building MNN with Vulkan in $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_VULKAN=ON \
    -DMNN_VULKAN_IMAGE=OFF \
    -DMNN_USE_SYSTEM_LIB=ON \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_BUILD_OPENCV=OFF \
    -DMNN_BUILD_AUDIO=OFF \
    -DMNN_BUILD_LLM=OFF \
    -DMNN_BUILD_DIFFUSION=OFF \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_ARM82=ON \
    -DMNN_OPENCL=OFF \
    -DMNN_USE_SSE=OFF \
    -DMNN_GPU_TIME_PROFILE=OFF \
    -DMNN_ISP_EMBED_SPIRV=ON

make -j$(nproc) MNN MNN_Vulkan MNN_Express

echo "Build complete!"
ls -lh OFF/libMNN.so
# Check for the symbol
nm -D OFF/libMNN.so | grep -i MNNVulkanFuseRegister && echo "Symbol MNNVulkanFuseRegister found in libMNN.so" || echo "Symbol NOT found!"
find -name "*.so"
