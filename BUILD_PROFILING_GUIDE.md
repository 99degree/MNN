# 🎯 MNN Profiling Build Guide

This guide provides step-by-step instructions to build MNN with profiling support and resolve the `MNN_GetSessionInfo` linker error.

## ⚡ Problem Solved

You encountered this error:
```
ld.lld: error: undefined symbol: MNN_GetSessionInfo
```

**Root Cause**: The function `MNN_GetSessionInfo` is not part of the standard MNN public API. However, MNN **does** provide profiling capabilities through `Interpreter::getSessionInfo()`.

## ✅ Solution: Build MNN with Profiling API

We've added a **C-compatible profiling API** that provides the `MNN_GetSessionInfo` function you need.

### Step 1: Build MNN with Profiling Support

```bash
# Clean any previous build
rm -rf build_profile install_profile

# Create build directory
mkdir -p build_profile && cd build_profile

# Configure with profiling enabled
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_GPU_TIME_PROFILE=ON \
    -DMNN_BUILD_TEST=ON \
    -DMNN_INTERNAL=ON \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_SEP_BUILD=OFF

# Build MNN
make -j$(nproc)

# Install (optional)
make install DESTDIR=../install_profile
```

### Step 2: Verify Profiling API is Available

After building, check that the profiling files are included:

```bash
# Check that our profiling files exist
ls -la ../include/MNN/MNNProfiling.h
ls -la ../source/core/MNNProfiling.cpp

# Check that the header is in the build
ls -la build_profile/include/MNN/MNNProfiling.h  # If installed
```

### Step 3: Use the Profiling API

#### Option A: C++ API (Recommended)

```cpp
#include <MNN/Interpreter.hpp>

// Create interpreter and session
auto interpreter = MNN::Interpreter::createFromFile("model.mnn");
MNN::ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
config.numThread = 4;
auto session = interpreter->createSession(config);

// Get profiling information
float memory = 0;
interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);

float flops = 0;
interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);

// Clean up
interpreter->releaseSession(session);
MNN::Interpreter::destroy(interpreter);
```

#### Option B: C API (For Your Existing Code)

```cpp
#include <MNN/MNNProfiling.h>

// Use the C wrapper functions
float memory = MNN_GetSessionMemory(interpreter, session);
float flops = MNN_GetSessionFlops(interpreter, session);
int threads = MNN_GetSessionThreadNumber(interpreter, session);

// Or use the direct function
float memoryDirect = 0;
bool success = MNN_GetSessionInfo(interpreter, session, MNN_SESSION_INFO_MEMORY, &memoryDirect);
```

### Step 4: Link Your Application

#### With CMake:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyApp)

# Find MNN (adjust paths as needed)
set(MNN_INCLUDE_DIR /path/to/mnn/include)
set(MNN_LIB_DIR /path/to/mnn/build_profile/lib)

# Create your executable
add_executable(my_app main.cpp)

# Link against MNN
target_link_libraries(my_app 
    ${MNN_LIB_DIR}/libMNN.so
    ${MNN_LIB_DIR}/libMNN_Express.so  # If using Express API
)

# Include directories
target_include_directories(my_app PRIVATE 
    ${MNN_INCLUDE_DIR}
)

# Enable C++11
target_compile_features(my_app PRIVATE cxx_std_11)
```

#### With Command Line:

```bash
g++ my_app.cpp \
    -I/path/to/mnn/include \
    -L/path/to/mnn/build_profile/lib \
    -lMNN \
    -o my_app
```

## 📋 What We've Added

### 1. C API Header (`include/MNN/MNNProfiling.h`)

```cpp
// Session information codes
typedef enum {
    MNN_SESSION_INFO_MEMORY = 0,     // Memory usage in MB
    MNN_SESSION_INFO_FLOPS = 1,       // Floating point operations in M
    MNN_SESSION_INFO_BACKENDS = 2,   // Backend types used
    MNN_SESSION_INFO_RESIZE_STATUS = 3, // Resize status
    MNN_SESSION_INFO_THREAD_NUMBER = 4, // Number of threads
} MNNSessionInfoCode;

// C API functions
MNN_PUBLIC bool MNN_GetSessionInfo(void* interpreter, void* session, MNNSessionInfoCode code, void* ptr);
MNN_PUBLIC float MNN_GetSessionMemory(void* interpreter, void* session);
MNN_PUBLIC float MNN_GetSessionFlops(void* interpreter, void* session);
MNN_PUBLIC int MNN_GetSessionThreadNumber(void* interpreter, void* session);
MNN_PUBLIC int MNN_GetSessionBackends(void* interpreter, void* session, int* backends, int maxBackends);
MNN_PUBLIC const char* MNN_GetVersion();
```

### 2. C API Implementation (`source/core/MNNProfiling.cpp`)

Provides the implementation that wraps the C++ `Interpreter::getSessionInfo()` method.

### 3. CMake Configuration Updated

Added `MNNProfiling.h` to the public headers list in `CMakeLists.txt`.

## 🔧 Build Options Explained

| Option | Purpose | Recommended Value |
|--------|---------|-------------------|
| `MNN_GPU_TIME_PROFILE` | Enable GPU time profiling (OpenCL/Vulkan) | **ON** |
| `MNN_INTERNAL` | Enable internal features and extended profiling | **ON** |
| `MNN_BUILD_TEST` | Build test utilities | **ON** |
| `MNN_BUILD_SHARED_LIBS` | Build shared libraries (.so files) | **ON** |
| `MNN_SEP_BUILD` | Separate build (don't enable for profiling) | **OFF** |

## 🎯 Quick Verification

To quickly verify that everything works:

```bash
# Build MNN with profiling
cd /path/to/mnn
./build_profiling.sh

# Check that libMNN.so contains the profiling symbols
cd build_profile/lib
nm -D libMNN.so | grep -i "MNN_GetSessionInfo\|MNN_GetSessionMemory\|MNN_GetSessionFlops"

# You should see output like:
# 0000000000123456 T MNN_GetSessionInfo
# 0000000000123460 T MNN_GetSessionMemory
# etc.
```

## 🛠️ Troubleshooting

### Problem: Still getting undefined symbol error

**Solution**: 
1. Make sure you're linking against the **newly built** libMNN.so
2. Verify the symbols are present: `nm -D libMNN.so | grep MNN_GetSessionInfo`
3. Check your include paths: `-I/path/to/mnn/include`
4. Check your library paths: `-L/path/to/mnn/build_profile/lib`

### Problem: CMake can't find MNN

**Solution**: 
1. Set the correct paths in your CMakeLists.txt
2. Or use `find_package(MNN REQUIRED)` if MNN is installed
3. Or add MNN as a subdirectory: `add_subdirectory(/path/to/mnn)`

### Problem: Build fails with compilation errors

**Solution**: 
1. Make sure you have all dependencies installed
2. Check the build log for specific errors
3. Try a clean build: `rm -rf build_profile && mkdir build_profile && cd build_profile && cmake ..`

## 📊 Example: Complete Profiling Workflow

```cpp
#include <MNN/Interpreter.hpp>
#include <MNN/MNNProfiling.h>
#include <iostream>

int main() {
    // 1. Load model
    auto interpreter = MNN::Interpreter::createFromFile("model.mnn");
    if (!interpreter) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // 2. Create session with profiling
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    auto session = interpreter->createSession(config);
    
    // 3. Get profiling information (C++ API)
    float memory = 0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);
    std::cout << "Memory: " << memory << " MB" << std::endl;
    
    // 4. Get profiling information (C API)
    float memory_c = MNN_GetSessionMemory(interpreter, session);
    std::cout << "Memory (C API): " << memory_c << " MB" << std::endl;
    
    // 5. Run inference
    auto error = interpreter->runSession(session);
    if (error != MNN::NO_ERROR) {
        std::cerr << "Inference failed: " << error << std::endl;
        return 1;
    }
    
    // 6. Clean up
    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    
    return 0;
}
```

## 📈 Available Metrics

| Metric | C++ API | C API | Description |
|--------|---------|-------|-------------|
| Memory Usage | `Interpreter::MEMORY` | `MNN_SESSION_INFO_MEMORY` | Memory in MB |
| FLOPS | `Interpreter::FLOPS` | `MNN_SESSION_INFO_FLOPS` | Floating point operations in M |
| Backend Types | `Interpreter::BACKENDS` | `MNN_SESSION_INFO_BACKENDS` | Backend enum values |
| Resize Status | `Interpreter::RESIZE_STATUS` | `MNN_SESSION_INFO_RESIZE_STATUS` | 0=ready, 1=need_malloc, 2=need_resize |
| Thread Count | `Interpreter::THREAD_NUMBER` | `MNN_SESSION_INFO_THREAD_NUMBER` | Number of threads |

## 🎉 Success Criteria

✅ **Build completes** without errors  
✅ **libMNN.so** contains `MNN_GetSessionInfo` symbol  
✅ **Your application links** successfully  
✅ **Profiling data** is accessible via API  

## 📞 Need More Help?

If you're still having issues:

1. **Check the build logs** in `build_profile/`
2. **Verify symbol export**: `nm -D libMNN.so | grep -i profile`
3. **Test with our example**: Build `example_profiling.cpp`
4. **Ask for help** with specific error messages

The profiling API is now fully integrated into MNN and ready to use! 🚀