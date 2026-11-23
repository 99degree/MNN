#!/data/data/com.termux/files/usr/bin/bash
set -e

# Android Vulkan build script for MNN (using Termux native clang)
# Usage: ./build_vulkan_android.sh [BUILD_DIR] [ANDROID_ABI] [ANDROID_API_LEVEL] [BUILD_CONVERTER]

BUILD_DIR="${1:-build_isp}"
ANDROID_ABI="${2:-arm64-v8a}"
ANDROID_API_LEVEL="${3:-android-21}"
BUILD_CONVERTER="${4:-ON}"

# Termux native clang targets aarch64-unknown-linux-android24 by default
# Extract API level from ANDROID_API_LEVEL (e.g., android-21 -> 21)
API_LEVEL="${ANDROID_API_LEVEL#android-}"

# Map ABI to target triple
if [ "$ANDROID_ABI" = "armeabi-v7a" ]; then
    TARGET_TRIPLE="armv7a-linux-androideabi${API_LEVEL}"
    ANDROID_STL="c++_static"
    ARM82="OFF"
elif [ "$ANDROID_ABI" = "arm64-v8a" ]; then
    TARGET_TRIPLE="aarch64-linux-android${API_LEVEL}"
    ANDROID_STL="c++_static"
    ARM82="ON"
elif [ "$ANDROID_ABI" = "x86" ]; then
    TARGET_TRIPLE="i686-linux-android${API_LEVEL}"
    ANDROID_STL="c++_static"
    ARM82="OFF"
elif [ "$ANDROID_ABI" = "x86_64" ]; then
    TARGET_TRIPLE="x86_64-linux-android${API_LEVEL}"
    ANDROID_STL="c++_static"
    ARM82="OFF"
else
    echo "Unsupported ABI: $ANDROID_ABI"
    exit 1
fi

cd "$(dirname "$0")"

echo "Building MNN with Vulkan for Android (Termux native clang)"
echo "  Build Dir: $BUILD_DIR"
echo "  ABI: $ANDROID_ABI"
echo "  Target: $TARGET_TRIPLE"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Use Termux native clang with explicit target
export CC="/data/data/com.termux/files/usr/bin/clang"
export CXX="/data/data/com.termux/files/usr/bin/clang++"
export AS="/data/data/com.termux/files/usr/bin/clang"
export AR="/data/data/com.termux/files/usr/bin/llvm-ar"
export RANLIB="/data/data/com.termux/files/usr/bin/llvm-ranlib"
export STRIP="/data/data/com.termux/files/usr/bin/llvm-strip"

# Set target flags via CFLAGS/CXXFLAGS - Termux clang needs -target
export CFLAGS="--target=$TARGET_TRIPLE -fPIC"
export CXXFLAGS="--target=$TARGET_TRIPLE -fPIC"

# Vulkan: source/backend/vulkan/CMakeLists.txt prefers the NDK's sysroot stub
# over whatever find_package(Vulkan) resolves on the host, because Termux ships
# libvulkan.so -> libvulkan.so.1 (SONAME libvulkan.so.1) and an Android device
# only has SONAME libvulkan.so. Without this every MNN_Vulkan/libMNNConvertDeps
# build gets `NEEDED libvulkan.so.1` and crashes at dlopen on device.
if [ -z "${ANDROID_NDK_ROOT:-}" ] && [ -d "$HOME/android-sdk/ndk" ]; then
    # Pick the newest installed NDK.
    export ANDROID_NDK_ROOT="$(ls -d "$HOME"/android-sdk/ndk/*/ 2>/dev/null | sort -V | tail -1)"
    export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT%/}"
fi

cmake .. \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_ASM_COMPILER="$AS" \
    -DCMAKE_AR="$AR" \
    -DCMAKE_RANLIB="$RANLIB" \
    -DCMAKE_STRIP="$STRIP" \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=21 \
    -DCMAKE_ANDROID_ARCH_ABI="$ANDROID_ABI" \
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