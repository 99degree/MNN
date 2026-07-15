# MNN Profiling Support

This guide explains how to build MNN with profiling support and use the profiling APIs to analyze model performance.

## 🚀 Quick Start

### 1. Build MNN with Profiling Support

```bash
# Method 1: Using the build script
./build_profiling.sh

# Method 2: Manual build
mkdir -p build_profile && cd build_profile
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_GPU_TIME_PROFILE=ON \
    -DMNN_BUILD_TEST=ON \
    -DMNN_INTERNAL=ON \
    -DMNN_BUILD_SHARED_LIBS=ON
make -j$(nproc)
```

### 2. Update Your Build Configuration

If you're already building MNN, add these flags to your CMake configuration:

```cmake
# In your CMakeLists.txt
option(MNN_ENABLE_PROFILING "Enable MNN profiling features" ON)
set(MNN_GPU_TIME_PROFILE ON)
set(MNN_INTERNAL ON)
set(MNN_BUILD_TEST ON)
```

## 📊 Available Profiling APIs

### C++ API (Recommended)

The main profiling interface is through `Interpreter::getSessionInfo()`:

```cpp
#include <MNN/Interpreter.hpp>

// Create interpreter and session
auto interpreter = MNN::Interpreter::createFromFile("model.mnn");
MNN::ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
config.numThread = 4;
auto session = interpreter->createSession(config);

// Get profiling information
float memoryMB = 0;
interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryMB);

float flops = 0;
interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);

int threadCount = 0;
interpreter->getSessionInfo(session, MNN::Interpreter::THREAD_NUMBER, &threadCount);
```

### C API (For Compatibility)

If you need a C interface, use the provided wrapper functions:

```cpp
#include <MNN/MNNProfiling.h>

// C API functions
bool MNN_GetSessionInfo(void* interpreter, void* session, MNNSessionInfoCode code, void* ptr);
float MNN_GetSessionMemory(void* interpreter, void* session);
float MNN_GetSessionFlops(void* interpreter, void* session);
int MNN_GetSessionThreadNumber(void* interpreter, void* session);
int MNN_GetSessionBackends(void* interpreter, void* session, int* backends, int maxBackends);
const char* MNN_GetVersion();
```

### Session Information Codes

| Code | Description | Data Type | Unit |
|------|-------------|-----------|------|
| `MEMORY` | Memory usage | `float*` | MB |
| `FLOPS` | Floating point operations | `float*` | M (million) |
| `BACKENDS` | Backend types used | `int*` | Backend enum |
| `RESIZE_STATUS` | Resize status | `int*` | - |
| `THREAD_NUMBER` | Number of threads | `int*` | - |

## 🎯 Usage Examples

### Basic Profiling

```cpp
#include <MNN/Interpreter.hpp>
#include <iostream>

int main() {
    auto interpreter = MNN::Interpreter::createFromFile("model.mnn");
    
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    
    auto session = interpreter->createSession(config);
    
    // Profile before execution
    float memoryBefore = 0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryBefore);
    
    // Run inference
    interpreter->runSession(session);
    
    // Profile after execution
    float memoryAfter = 0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryAfter);
    
    std::cout << "Memory delta: " << (memoryAfter - memoryBefore) << " MB" << std::endl;
    
    // Clean up
    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    
    return 0;
}
```

### Advanced Profiling with Callbacks

```cpp
#include <MNN/Interpreter.hpp>
#include <iostream>

int main() {
    auto interpreter = MNN::Interpreter::createFromFile("model.mnn");
    
    // Enable debug mode for callbacks
    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);
    
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 2;
    
    auto session = interpreter->createSession(config);
    
    // Define profiling callbacks
    MNN::TensorCallBackWithInfo beforeCallback = [](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        std::cout << "Before: " << info->name() 
                  << " (type: " << info->type() 
                  << ", flops: " << info->flops() << " M)" << std::endl;
        return true;
    };
    
    MNN::TensorCallBackWithInfo afterCallback = [](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        std::cout << "After: " << info->name() << std::endl;
        return true;
    };
    
    // Run with callbacks
    interpreter->runSessionWithCallBackInfo(session, beforeCallback, afterCallback, true);
    
    // Clean up
    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    
    return 0;
}
```

### GPU Profiling (OpenCL/Vulkan)

For GPU backends, enable time profiling:

```cpp
#include <MNN/Interpreter.hpp>

int main() {
    auto interpreter = MNN::Interpreter::createFromFile("model.mnn");
    
    // Use Vulkan backend with profiling
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_VULKAN;  // or MNN_FORWARD_OPENCL
    config.numThread = 4;
    
    auto session = interpreter->createSession(config);
    
    // Get profiling information
    float memory = 0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);
    
    float flops = 0;
    interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    
    // Run inference
    interpreter->runSession(session);
    
    // Note: GPU time profiling data is available through the same API
    // when MNN_GPU_TIME_PROFILE is enabled during build
    
    // Clean up
    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    
    return 0;
}
```

## 🔧 Build Configuration Options

### CMake Options for Profiling

| Option | Description | Default | Recommended |
|--------|-------------|---------|-------------|
| `MNN_GPU_TIME_PROFILE` | Enable GPU time profiling (OpenCL/Vulkan) | OFF | ON |
| `MNN_INTERNAL` | Enable internal features and extended profiling | OFF | ON |
| `MNN_BUILD_TEST` | Build test utilities | OFF | ON |
| `MNN_ENABLE_PROFILING` | Enable profiling features (custom) | OFF | ON |

### Example CMake Configuration

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyMNNProject)

# Find MNN
find_package(MNN REQUIRED)

# Or specify paths directly
set(MNN_INCLUDE_DIRS /path/to/mnn/include)
set(MNN_LIBRARIES /path/to/mnn/lib/libMNN.so)

# Your target
add_executable(my_app main.cpp)
target_link_libraries(my_app ${MNN_LIBRARIES})
target_include_directories(my_app PRIVATE ${MNN_INCLUDE_DIRS})
```

## 🛠️ Troubleshooting

### Linker Error: Undefined Symbol MNN_GetSessionInfo

**Problem**: The linker cannot find `MNN_GetSessionInfo`.

**Solution**: 
1. Make sure you're linking against the MNN library built with profiling support
2. Use the C++ API instead: `Interpreter::getSessionInfo()`
3. If you need the C API, ensure you're using the correct header: `#include <MNN/MNNProfiling.h>`

### No Profiling Data Available

**Problem**: `getSessionInfo()` returns false or zero values.

**Solution**:
1. Ensure you've created a session before calling `getSessionInfo()`
2. Check that the session pointer is valid
3. Verify that you're using the correct session info code

### GPU Profiling Not Working

**Problem**: GPU profiling data is not available.

**Solution**:
1. Make sure you built MNN with `-DMNN_GPU_TIME_PROFILE=ON`
2. Use a GPU backend (`MNN_FORWARD_OPENCL` or `MNN_FORWARD_VULKAN`)
3. Check that your GPU drivers are properly installed

## 📈 Performance Metrics

The profiling API provides several key metrics:

### Memory Usage
- **What it measures**: Total memory used by the session
- **Unit**: Megabytes (MB)
- **When to use**: To understand memory consumption of your model

### FLOPS
- **What it measures**: Total floating point operations
- **Unit**: Million FLOPS (M)
- **When to use**: To estimate computational complexity

### Backend Information
- **What it measures**: Which backends are being used
- **Unit**: Backend enum values
- **When to use**: To verify backend selection

### Thread Count
- **What it measures**: Number of threads used by the session
- **Unit**: Integer count
- **When to use**: To verify parallelization

### Resize Status
- **What it measures**: Current resize state
- **Values**: 0=ready, 1=need_malloc, 2=need_resize
- **When to use**: To debug memory allocation issues

## 🔄 Integration with Existing Code

If you have existing code that expects `MNN_GetSessionInfo`, you can either:

### Option 1: Update to Use C++ API (Recommended)
```cpp
// Old code (may not work)
// MNN_GetSessionInfo(session, code, ptr);

// New code (recommended)
interpreter->getSessionInfo(session, static_cast<MNN::Interpreter::SessionInfoCode>(code), ptr);
```

### Option 2: Use the C Wrapper
```cpp
// Include the profiling header
#include <MNN/MNNProfiling.h>

// Use the C wrapper
MNN_GetSessionInfo(interpreter, session, MNN_SESSION_INFO_MEMORY, &memory);
```

## 📁 Files Added

This profiling setup adds the following files to MNN:

1. **`include/MNN/MNNProfiling.h`** - C API header for profiling
2. **`source/core/MNNProfiling.cpp`** - C API implementation
3. **`example_profiling.cpp`** - Example demonstrating profiling usage
4. **`build_profiling.sh`** - Build script for profiling-enabled MNN
5. **`CMakeLists_profiling.txt`** - CMake configuration for profiling
6. **`PROFILING.md`** - This documentation

## 🎓 Best Practices

1. **Always check return values**: The profiling functions may fail, so always check the return value.

2. **Use appropriate data types**: Make sure the pointer you pass matches the expected data type for the session info code.

3. **Profile at the right time**: Memory usage may change during execution, so profile before and after critical operations.

4. **Clean up resources**: Always release sessions and destroy interpreters when done.

5. **Consider performance impact**: Profiling may have a small performance overhead, so disable it in production builds.

## 🔗 Related Documentation

- [MNN Main Documentation](https://github.com/alibaba/MNN)
- [Interpreter API Documentation](docs/cpp/Interpreter.md)
- [Backend Configuration](docs/cpp/Backend.md)

## 📞 Support

If you encounter issues with profiling:

1. Check that you've built MNN with the correct flags
2. Verify that your model loads successfully
3. Ensure you're using valid session pointers
4. Check the return values of all profiling functions

For additional help, please refer to the [MNN GitHub repository](https://github.com/alibaba/MNN) or open an issue with details about your problem.