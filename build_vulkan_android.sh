#!/data/data/com.termux/files/usr/bin/bash
set -e

# Android Vulkan build script for MNN (using Android NDK)
# Usage: ./build_vulkan_android.sh [BUILD_DIR] [ANDROID_ABI] [ANDROID_API_LEVEL] [BUILD_CONVERTER] [ANDROID_NDK]

BUILD_DIR="${1:-build_isp}"
ANDROID_ABI="${2:-arm64-v8a}"
ANDROID_API_LEVEL="${3:-android-21}"
BUILD_CONVERTER="${4:-ON}"
ANDROID_NDK="${5:-$ANDROID_NDK}"

if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK environment variable not set and not provided as argument"
    echo "Usage: $0 [BUILD_DIR] [ANDROID_ABI] [ANDROID_API_LEVEL] [BUILD_CONVERTER] [ANDROID_NDK]"
    echo "Example: $0 build_isp arm64-v8a android-21 ON ~/android-sdk/ndk/29.0.14206865"
    exit 1
fi

# Expand tilde in NDK path
ANDROID_NDK="${ANDROID_NDK/#\~/$HOME}"

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: NDK path does not exist: $ANDROID_NDK"
    exit 1
fi

TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "Error: Android toolchain file not found: $TOOLCHAIN_FILE"
    exit 1
fi

cd "$(dirname "$0")"

echo "Building MNN with Vulkan for Android"
echo "  Build Dir: $BUILD_DIR"
echo "  ABI: $ANDROID_ABI"
echo "  API Level: $ANDROID_API_LEVEL"
echo "  NDK: $ANDROID_NDK"

# Set STL based on ABI
if [ "$ANDROID_ABI" = "armeabi-v7a" ]; then
    ANDROID_STL="c++_static"
    ARM82="OFF"
else
    ANDROID_STL="c++_static"
    ARM82="ON"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Use Android NDK toolchain with legacy toolchain file (newer toolchain requires newer CMake)
export ANDROID_USE_LEGACY_TOOLCHAIN_FILE=ON

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_STL="$ANDROID_STL" \
    -DANDROID_NATIVE_API_LEVEL="$ANDROID_API_LEVEL" \
    -DANDROID_TOOLCHAIN=clang \
    -DMNN_VULKAN=ON \
    -DMNN_VULKAN_IMAGE=OFF \
    -DMNN_USE_SYSTEM_LIB=ON \
    -DMNN_SEP_BUILD=ON \
    -DMNN_BUILD_OPENCV=OFF \
    -DMNN_BUILD_AUDIO=OFF \
    -DMNN_BUILD_LLM=OFF \
    -DMNN_BUILD_DIFFUSION=OFF \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_ARM82=$ARM82 \
    -DMNN_OPENCL=OFF \
    -DMNN_USE_SSE=OFF \
    -DMNN_GPU_TIME_PROFILE=ON \
    -DMNN_ISP_EMBED_SPIRV=ON \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
    -DMNN_USE_LOGCAT=ON \
    -DMNN_BUILD_CONVERTER=$BUILD_CONVERTER \
    -DNATIVE_LIBRARY_OUTPUT=. \
    -DNATIVE_INCLUDE_OUTPUT=.

make -j$(nproc) MNN MNN_Vulkan MNN_Express

if [ "$BUILD_CONVERTER" = "ON" ]; then
    echo "Building MNNConvertDeps..."
    make -j$(nproc) MNNConvertDeps
fi

echo "Build complete!"
ls -lh *.so 2>/dev/null || true

# Check for the symbol in libMNN_Vulkan.so (since SEP_BUILD=ON)
if [ -f "source/backend/vulkan/libMNN_Vulkan.so" ]; then
    if nm -D source/backend/vulkan/libMNN_Vulkan.so 2>/dev/null | grep -q ' T MNNVulkanFuseRegister'; then
        echo "Symbol MNNVulkanFuseRegister found in libMNN_Vulkan.so"
    else
        echo "Symbol NOT found in libMNN_Vulkan.so"
    fi
elif [ -f "libMNN_Vulkan.so" ]; then
    if nm -D libMNN_Vulkan.so 2>/dev/null | grep -q ' T MNNVulkanFuseRegister'; then
        echo "Symbol MNNVulkanFuseRegister found in libMNN_Vulkan.so"
    else
        echo "Symbol NOT found in libMNN_Vulkan.so"
    fi
fi

find . -name "*.so" -exec ls -lh {} \;