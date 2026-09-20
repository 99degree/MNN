#!/bin/bash
set -e

BUILD_DIR="${1:-build_isp}"
BUILD_CONVERTER="${2:-ON}"

cd "$(dirname "$0")"

echo "Building MNN with Vulkan in $BUILD_DIR (CONVERTER=$BUILD_CONVERTER)"

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
    -DMNN_GPU_TIME_PROFILE=ON \
    -DMNN_ISP_EMBED_SPIRV=OFF \
    -DMNN_BUILD_CONVERTER=$BUILD_CONVERTER

make -j$(nproc) MNN MNN_Vulkan MNN_Express

if [ "$BUILD_CONVERTER" = "ON" ]; then
    echo "Building MNNConvertDeps..."
    make -j$(nproc) MNNConvertDeps
fi

echo "Build complete!"
ls -lh OFF/libMNN.so

# Check for the symbol (in libMNN.so when SEP_BUILD=OFF, or libMNN_Vulkan.so when SEP_BUILD=ON)
if nm -D OFF/libMNN.so | grep -q ' T MNNVulkanFuseRegister'; then
    echo "Symbol MNNVulkanFuseRegister found in libMNN.so"
elif nm -D source/backend/vulkan/OFF/libMNN_Vulkan.so 2>/dev/null | grep -q ' T MNNVulkanFuseRegister'; then
    echo "Symbol MNNVulkanFuseRegister found in libMNN_Vulkan.so"
else
    echo "Symbol NOT found!"
fi
if [ "$BUILD_CONVERTER" = "ON" ]; then
    find . -name "libMNNConvertDeps.so" -exec ls -lh {} \;
fi
find . -name "*.so"
