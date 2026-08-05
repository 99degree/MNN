# Vulkan Zero-Copy & ISP Extensions: Architecture Analysis

## 1. Build Topology — What Goes Where

The CI builds with **`MNN_SEP_BUILD=ON`** (the default). Each backend is a **separate `.so`**.

```
libMNN.so            ← Core inference engine, CPU backend, Express API, Tensor API
libMNN_Vulkan.so     ← ALL Vulkan code (buffer, image, ISP extensions, zero-copy)
libMNN_CL.so         ← OpenCL backend
libMNN_Express.so    ← Express dynamic graph
libMNNConvertDeps.so ← Converter tooling
libMNNOpenCV.so      ← OpenCV integration
libMNNAudio.so       ← Audio support
```

**Key point:** `VulkanISPExtensions.cpp` and the zero-copy `VulkanBuffer`/`VulkanBackend` code are both in `libMNN_Vulkan.so`, never in `libMNN.so`.

Source: `CMakeLists.txt:693-701`, `source/backend/vulkan/CMakeLists.txt:23-44`

---

## 2. Two Separate Code Paths — They Don't Talk to Each Other

### Path A: Zero-Copy External Memory (buffer backend only)

```
Client code                    MNN internal
───────────                    ────────────
tensor->setDevicePtr(          Tensor.hpp:310 → Tensor.cpp:498
  ahb_handle,                  sets mBuffer.flags = MNN_MEMORY_AHARDWAREBUFFER (14)
  MNN_MEMORY_AHARDWAREBUFFER)  sets mBuffer.device = (uint64_t)ahb_handle
                               ↓
                               [on forward/onAcquire]
                               ↓
                               VulkanBackend::onAcquire()       ← buffer/backend/VulkanBackend.cpp:241
                               ↓ checks flags == MNN_MEMORY_AHARDWAREBUFFER
                               ↓ handle > 1024? → createExternalAHB()
                               ↓                → createExternal(fd)
                               VulkanBuffer(AHardwareBuffer*)   ← component/VulkanBuffer.cpp:148
                               ↓ VkExternalMemoryBufferCreateInfo
                               ↓ VkImportAndroidHardwareBufferInfoANDROID
                               ↓ gpu import, no CPU memcpy
```

**Trigger:** `Tensor::setDevicePtr(handle, MNN_MEMORY_AHARDWAREBUFFER)` — a **public API** in `Tensor.hpp:310` and `Expr.hpp:119`.

**Guard:** This path only exists when `MNN_VULKAN_IMAGE=OFF` (buffer mode). The CI default is `MNN_VULKAN_IMAGE=ON` (image mode), where this code is **not compiled**.

### Path B: ISP Extensions C API (stubs)

```
Client code                    MNN internal
───────────                    ────────────
MNNVulkanHotSwapConstBuffer()  VulkanISPExtensions.cpp:65
MNNVulkanQueryOptimalWorkgroup  VulkanISPExtensions.cpp:38
MNNVulkanSetSessionWorkgroup    VulkanISPExtensions.cpp:50
MNNVulkanSetWorkgroupPreset     VulkanISPExtensions.cpp:56
                               ↓
                               All stubs / TODO. Store data in
                               global maps but never feed it
                               into any Vulkan pipeline.
```

**Current state:** These are **dead code stubs**. They store values in `std::map` globals but nothing reads them back.

---

## 3. How a Client Test Links to libMNN.so and Achieves Zero-Copy

### 3a. Linking Model

When `MNN_SEP_BUILD=ON`, the client only links against **`libMNN.so`**. The Vulkan backend is loaded at runtime via **static init self-registration**:

```
libMNN_Vulkan.so loaded by dynamic linker
  → static init calls MNNInsertExtraRuntimeCreator(
        MNN_FORWARD_VULKAN, new VulkanRuntimeCreator, true)
  → registers creator in libMNN.so's global map
  → no explicit dlopen needed from client code
```

The client must ensure `libMNN_Vulkan.so` is on the library search path:
- **Android NDK / CMake:** The `.so` files are in `jniLibs/<abi>/` — automatically included in APK
- **Linux desktop:** `LD_LIBRARY_PATH` must include the directory with all `.so` files
- **CMake linking:** Only `-lMNN` is needed; Vulkan backend self-registers

### 3b. Build Requirements (Critical)

The CI builds image mode by default (`MNN_VULKAN_IMAGE=ON`). For zero-copy, you **must** build with buffer mode:

```bash
# Build MNN with zero-copy support:
cmake .. \
  -DMNN_SEP_BUILD=ON \
  -DMNN_VULKAN=ON \
  -DMNN_VULKAN_IMAGE=OFF \       # ← CRITICAL: buffer mode for zero-copy path
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=ON
make -j$(nproc)
```

This produces both `libMNN.so` and `libMNN_Vulkan.so` (with the buffer backend compiled in).

### 3c. Complete Test Program (Android AHardwareBuffer)

```cpp
// zero_copy_test.cpp
// Build: aarch64-linux-android34-clang++ -std=c++17 -o zero_copy_test \
//   zero_copy_test.cpp -lMNN -llog -landroid -lEGL -lGLESv2 -ldl

#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <android/hardware_buffer.h>
#include <dlfcn.h>

using namespace MNN;
using namespace MNN::Express;

// --- AHardwareBuffer helper (via dlopen to avoid link-time dep) ---
typedef int (*PFN_AHardwareBuffer_allocate)(
    const AHardwareBuffer_Desc*, AHardwareBuffer**);
typedef void (*PFN_AHardwareBuffer_release)(AHardwareBuffer*);
typedef int (*PFN_AHardwareBuffer_lock)(AHardwareBuffer*, uint64_t,
    int32_t, const ARect*, void**);
typedef int (*PFN_AHardwareBuffer_unlock)(AHardwareBuffer*, int32_t*);

static PFN_AHardwareBuffer_allocate pAlloc = nullptr;
static PFN_AHardwareBuffer_release  pRelease = nullptr;
static PFN_AHardwareBuffer_lock     pLock = nullptr;
static PFN_AHardwareBuffer_unlock   pUnlock = nullptr;

static void loadAhbFuncs() {
    void* h = dlopen("libandroid.so", RTLD_NOW);
    pAlloc   = (PFN_AHardwareBuffer_allocate)dlsym(h, "AHardwareBuffer_allocate");
    pRelease = (PFN_AHardwareBuffer_release)dlsym(h, "AHardwareBuffer_release");
    pLock    = (PFN_AHardwareBuffer_lock)dlsym(h, "AHardwareBuffer_lock");
    pUnlock  = (PFN_AHardwareBuffer_unlock)dlsym(h, "AHardwareBuffer_unlock");
}

static AHardwareBuffer* createAhb(int w, int h, int format, void* data) {
    AHardwareBuffer_Desc desc = {};
    desc.width  = w;
    desc.height = h;
    desc.layers = 1;
    desc.format = format;  // AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM
    desc.usage  = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
                  AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;

    AHardwareBuffer* ahb = nullptr;
    pAlloc(&desc, &ahb);
    if (data && ahb) {
        ARect rect = {0, 0, w, h};
        void* ptr = nullptr;
        pLock(ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, &rect, &ptr);
        memcpy(ptr, data, w * h * 4);
        pUnlock(ahb, nullptr);
    }
    return ahb;
}

// --- Main test flow ---
int main() {
    loadAhbFuncs();

    const int W = 1280, C = 3, H = 720;

    // 1. Create executor with Vulkan backend
    RuntimeConfig config;
    config.type = MNN_FORWARD_VULKAN;
    auto executor = Executor::newExecutor(config);

    // 2. Build a simple model (transpose in this case)
    auto net = _Input({1, C, H, W}, NCHW, halide_type_of<float>(), executor);
    net->setName("input");
    auto out = _Transpose(net, {0, 1, 3, 2});
    out->setName("output");
    auto exe = Executor::extract({net}, {out});

    // 3. Create an AHardwareBuffer and fill it with test data
    float inputData[W * C * H];
    for (int i = 0; i < W * C * H; i++) inputData[i] = (float)(rand() % 255);
    AHardwareBuffer* inputAhb = createAhb(W, C * H,
        AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM, nullptr);

    // 4. Bind AHardwareBuffer to input tensor — THIS IS THE ZERO-COPY PATH
    //    Tensor::setDevicePtr stores ahb pointer + MNN_MEMORY_AHARDWAREBUFFER flag
    //    VulkanBackend::onAcquire() detects the flag and calls
    //    VulkanBuffer::createExternalAHB() which does VkImportAndroidHardwareBufferInfoANDROID
    volatile uint64_t ahbPtr = (uint64_t)inputAhb;
    net->setDevicePtr((void*)ahbPtr, MNN_MEMORY_AHARDWAREBUFFER);

    // 5. Run inference — Vulkan imports AHB into GPU memory (no CPU copy)
    auto outputs = exe->onForward({net});

    // 6. Get output — also zero-copy via AHardwareBuffer
    AHardwareBuffer* outputAhb = createAhb(W, C * H,
        AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM, nullptr);
    volatile uint64_t outPtr = (uint64_t)outputAhb;
    outputs[0]->copyToDevicePtr((void*)outPtr, MNN_MEMORY_AHARDWAREBUFFER);

    // 7. Read back from output AHB to verify
    void* resultPtr = nullptr;
    ARect rect = {0, 0, W, C * H};
    pLock(outputAhb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, &rect, &resultPtr);
    // ... validate resultPtr ...
    pUnlock(outputAhb, nullptr);

    pRelease(inputAhb);
    pRelease(outputAhb);
    return 0;
}
```

### 3d. CMakeLists.txt for the test

```cmake
cmake_minimum_required(VERSION 3.18)
project(zero_copy_test)

add_executable(zero_copy_test zero_copy_test.cpp)

# Link only libMNN.so — the Vulkan backend self-registers at runtime
target_link_libraries(zero_copy_test
    MNN         # Core library (loads libMNN_Vulkan.so at runtime)
    log         # Android log
    android     # AHardwareBuffer API
    EGL GLESv2  # If your model needs GL interop
    dl          # dlopen for AHardwareBuffer functions
)

# Ensure libMNN_Vulkan.so is in the same APK jniLibs/<abi>/ directory
```

### 3e. Existing Tests (updated)

`test/sharedmem/AhardWareBufferTest.cpp` now supports **both OpenCL and Vulkan**:

- **`AhardWareBufferTest`** ("sharedmem/AhardWareBuffer"): Runs when `--forwardtype` is OpenCL or Vulkan.
  Tests RGBA and YUV420 zero-copy with CPU reference comparison.
- **`VulkanZeroCopyTest`** ("sharedmem/VulkanZeroCopy"): **Self-contained** — creates its own
  Vulkan executor internally via `Executor::newExecutor(MNN_FORWARD_VULKAN, ...)`.
  Tests RGBA zero-copy with CPU reference and speed benchmark.
  Gracefully skips if `libMNN_Vulkan.so` is not loadable.

To run:
```bash
# Via test harness (OpenCL):
./run_test.out -i sharedmem/AhardWareBuffer --forwardtype=4  # MNN_FORWARD_OPENCL=4

# Via test harness (Vulkan, when built with buffer mode):
./run_test.out -i sharedmem/AhardWareBuffer --forwardtype=3  # MNN_FORWARD_VULKAN=3

# Self-contained Vulkan test (no --forwardtype needed):
./run_test.out -i sharedmem/VulkanZeroCopy
```

**Build requirement for Vulkan tests:** `-DMNN_VULKAN_IMAGE=OFF` (buffer mode). The Vulkan
image backend does **not** implement `MNN_MEMORY_AHARDWAREBUFFER` in its `onAcquire()`.

---

## 4. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Application                       │
│                                                                 │
│  1. Create AHardwareBuffer (from camera HAL / Gralloc)         │
│  2. input->setDevicePtr(ahb, MNN_MEMORY_AHARDWAREBUFFER)       │
│     └→ Tensor::mBuffer.flags = 14, .device = (uint64_t)ahb     │
│  3. executor->onForward({input})                                │
│     │                                                           │
│  ┌──▼──────────────────────────────────────────────────────────┐│
│  │                  libMNN.so (core)                            ││
│  │  Calls VulkanBackend::onAcquire() from libMNN_Vulkan.so    ││
│  └──┬──────────────────────────────────────────────────────────┘│
│     │                                                           │
│  ┌──▼──────────────────────────────────────────────────────────┐│
│  │              libMNN_Vulkan.so (auto-loaded)                 ││
│  │  VulkanBackend::onAcquire()                                 ││
│  │    ↓ flags == MNN_MEMORY_AHARDWAREBUFFER                    ││
│  │    ↓ VulkanBuffer::createExternalAHB(pool, ahb, size)       ││
│  │    ↓ VkExternalMemoryBufferCreateInfo                       ││
│  │    ↓ VkImportAndroidHardwareBufferInfoANDROID               ││
│  │    ↓ vkAllocateMemory + vkBindBufferMemory                  ││
│  │    → Zero-copy: GPU reads AHB directly                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  4. output->copyToDevicePtr(out_ahb, MNN_MEMORY_AHARDWAREBUFFER)│
│     └→ GPU writes result directly into output AHB              │
│  5. Read output AHB from CPU (no memcpy from GPU)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Can the Image Backend Also Support Zero-Copy AHB?

**Yes.** But the two backends store data differently, so the implementation differs.

### 5a. Why the Two Backends Are Different

```
Buffer backend (MNN_VULKAN_IMAGE=OFF):
  Tensor data → VkBuffer (linear, flat memory)
  Compute shaders → read/write VkBuffer (storage buffer)
  AHB import → VkBuffer directly → TRUE zero-copy ✓

Image backend (MNN_VULKAN_IMAGE=ON, default):
  Tensor data → VulkanTensor → VkImage (tiled, R32G32B32A32_SFLOAT)
  Compute shaders → read/write VkImage (storage image)
  AHB import → ??? (format/layout mismatch)
```

The image backend uses VkImage with `VK_FORMAT_R32G32B32A32_SFLOAT` (or FP16) in MNN's
NC4HW4 tiled layout. An AHardwareBuffer is typically `R8G8B8A8_UNORM` in linear layout.
You cannot import an AHB directly as the VkImage the image backend's shaders expect.

### 5b. Two Approaches for Image Backend AHB Support

#### Approach 1: Import AHB as VkBuffer + GPU-side bufferToImage (recommended)

```
AHB (RGBA8 linear)
  │
  ├─[VulkanBuffer::createExternalAHB]─► VkBuffer (zero-copy import, GPU-side)
  │                                          │
  │                                          ▼
  │                                    [VulkanImageConverter::encodeBufferToTensor]
  │                                          │
  │                                          ▼
  │                                    VkImage (R32G32B32A32_SFLOAT, NC4HW4 tiled)
  │                                          │
  └─[no CPU memcpy anywhere]────────────────► Compute shaders operate on VkImage
```

**This is "zero CPU-copy" — the data never touches CPU memory after AHB creation.**
There IS a GPU-side VkBuffer→VkImage copy, but it stays entirely on the GPU die
and is ~10-100x faster than a CPU→GPU PCIe/UMA transfer.

**What needs to change in `image/backend/VulkanBackend.cpp`:**

```cpp
// 1. onAcquire: add AHB check (same pattern as buffer backend)
Backend::MemObj* VulkanBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    auto MTensor = const_cast<Tensor*>(tensor);

    // NEW: detect MNN_MEMORY_AHARDWAREBUFFER
    if (MNN_MEMORY_AHARDWAREBUFFER == MTensor->buffer().flags) {
        int64_t handle = (int64_t)MTensor->buffer().device;
        auto* shared = static_cast<VulkanExternalMemRelease*>(
            TensorUtils::getSharedMem(MTensor));
        if (nullptr == shared || shared->handle() != handle) {
            // Import AHB as VkBuffer (reuse existing VulkanBuffer API)
            auto alignSize = VulkanTensor::getAlignSize(MTensor) * sizeof(float);
            std::shared_ptr<VulkanBuffer> extBuf =
                VulkanBuffer::createExternalAHB(getMemoryPool(),
                    (AHardwareBuffer*)handle, alignSize);
            shared = new VulkanExternalMemRelease(extBuf, handle);
            TensorUtils::setSharedMem(MTensor, shared);
        }
    }

    // 2. Still create VulkanTensor for compute shaders
    auto format = _getFormat(tensor->getType());
    auto newBuffer = std::make_shared<VulkanTensor>(
        MTensor, format, getMemoryPool(), device().proty().limits);
    MTensor->buffer().device = (uint64_t)(newBuffer.get());
    return new VulkanMemRelease(newBuffer);
}

// 3. onCopyBuffer: detect AHB source, use imported VkBuffer as source
//    for encodeBufferToTensor() instead of mmap+memcpy from CPU
void VulkanBackend::onCopyBuffer(const Tensor* src, const Tensor* dst) const {
    if (src->host<float>() != nullptr) {
        // ... existing host→gpu path (mmap + memcpy + bufferToImage)
    } else if (dst->host<void>() != nullptr) {
        // ... existing gpu→host path
    } else {
        // Check for AHB external memory
        auto* shared = static_cast<VulkanExternalMemRelease*>(
            TensorUtils::getSharedMem(src));
        if (shared && shared->buffer()) {
            // AHB VkBuffer → VkImage (GPU-side, no CPU involvement)
            auto vkTensor = reinterpret_cast<VulkanTensor*>(dst->deviceId());
            // Use VulkanImageConverter to encode imported VkBuffer → VkImage
            // (same pipeline as host→gpu but with different source buffer)
            ...
        } else {
            // ... existing device→device path
        }
    }
}
```

#### Approach 2: Import AHB directly as VkImage (not recommended)

```
AHB → VkImportAndroidHardwareBufferInfoANDROID → VkImage (RGBA8_UNORM)
                                                        │
                                                        ▼
                                            Format mismatch! Shaders expect
                                            R32G32B32A32_SFLOAT
```

Would require:
- `VK_ANDROID_external_memory_android_hardware_buffer` extension
- VkImage created with AHB-compatible format
- A conversion compute shader to handle RGBA8→RGBA32F
- Changes to VulkanImage class to support external memory
- **More invasive, less practical**

### 5c. What Changes in the CMake Build

With Approach 1, **no CMake changes are needed**. The VulkanBuffer API (`createExternalAHB`)
already lives in `component/VulkanBuffer.cpp`, which is compiled in BOTH image and buffer
builds (both globs include `component/*`).

The only new code goes into `image/backend/VulkanBackend.cpp`, which is always compiled
when `MNN_VULKAN_IMAGE=ON`.

### 5d. Result After the Change

| Build mode | AHB zero-copy | CPU memcpy | GPU copy | Notes |
|---|---|---|---|---|
| `MNN_VULKAN_IMAGE=OFF` (buffer) | True zero-copy | None | None | VkBuffer→compute |
| `MNN_VULKAN_IMAGE=ON` (image) | **Zero CPU-copy** | **None** | VkBuffer→VkImage | GPU-side only |
| Before change (image) | Not supported | Full CPU→CPU | CPU→GPU | Current default |

### 5e. Impact on SEP_BUILD

`MNN_SEP_BUILD` (ON vs OFF) is **irrelevant** to zero-copy support. Both image and buffer
backends live in `libMNN_Vulkan.so` when `MNN_SEP_BUILD=ON`, or in `libMNN.so` when OFF.
The `VulkanBuffer::createExternalAHB` API is available in both cases.

The only thing that matters is: does the Vulkan backend's `onAcquire()` handle
`MNN_MEMORY_AHARDWAREBUFFER`? Currently only the buffer backend does.
**Adding it to the image backend removes the last restriction.**

---

## 6. Comparison with OpenCL AHB Flow

OpenCL already has full AHB zero-copy support. Comparing the two backends reveals the
pattern the Vulkan image backend should follow.

### 6a. OpenCL AHB Data Flow (already working)

```
┌──────────────────────────────────────────────────────────────────────┐
│  onAcquire (memory import)                                           │
│                                                                      │
│  AHardwareBuffer* ahb                                                │
│    │                                                                 │
│    ├── Mali: cl::Buffer(ctx, flags, CL_IMPORT_TYPE_ARM,              │
│    │         CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM, ahb)        │
│    │         → zero-copy import as cl::Buffer                        │
│    │                                                                 │
│    └── Adreno: cl::Buffer(ctx, CL_MEM_USE_HOST_PTR |                │
│              CL_MEM_EXT_HOST_PTR_QCOM, &ahb_mem)                     │
│              → zero-copy import as cl::Buffer                        │
│                                                                      │
│  Wrapped in CLSharedMemReleaseBuffer(handle, clBuffer)               │
│  Stored via TensorUtils::setSharedMem()                              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  copyToDevice (AHB → MNN internal format)                           │
│                                                                      │
│  1. detect MNN_MEMORY_AHARDWAREBUFFER in srcTensor->buffer().flags  │
│  2. _allocHostBuffer() creates cl::Buffer from AHB (step above)     │
│  3. convertToDevice() → convertBetweenAHDandCLmem()                 │
│     ├── AHB cl::Buffer  ──[gl_to_cl kernel]──►  MNN cl::Buffer     │
│     └── AHB cl::Buffer  ──[yuv_to_cl kernel]──► MNN cl::Buffer     │
│        (GPU-side format conversion, no CPU memcpy)                   │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  copyFromDevice (MNN internal format → AHB)                         │
│                                                                      │
│  1. detect MNN_MEMORY_AHARDWAREBUFFER in dstTensor->buffer().flags  │
│  2. convertFromDevice() → convertBetweenAHDandCLmem()               │
│     ├── MNN cl::Buffer   ──[cl_to_gl kernel]──►  AHB cl::Buffer    │
│     └── MNN cl::Buffer   ──[cl_to_yuv kernel]──► AHB cl::Buffer    │
│        (GPU-side format conversion, no CPU memcpy)                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Key source files:**
- `opencl/core/OpenCLBackend.cpp:754-781` — `_allocHostBuffer`: AHB → cl::Buffer import
- `opencl/core/OpenCLBackend.cpp:960-966` — `convertToDevice`: routes to AHD conversion
- `opencl/core/OpenCLBackend.cpp:853-858` — `convertFromDevice`: routes to AHD conversion
- `opencl/core/BufferConvertor.cpp:575-669` — `convertBetweenAHDandCLmem`: GPU kernel dispatch
- `opencl/core/OpenCLBackend.hpp:270-288` — `CLSharedMemReleaseBuffer`: wrapper class

**Supported formats:** RGBA8 (`gl_to_cl`/`cl_to_gl`) and YUV420 (`yuv_to_cl`/`cl_to_yuv`).

### 6b. Side-by-Side Comparison

```
                     OpenCL (working)              Vulkan image backend (proposed)
                     ─────────────────             ──────────────────────────────
onAcquire:           AHB → cl::Buffer              AHB → VkBuffer
                     (vendor import ext)           (VK_EXTERNAL_MEMORY_HANDLE_TYPE_
                                                   ANDROID_HARDWARE_BUFFER_BIT_ANDROID)
                                                   Reuses VulkanBuffer::createExternalAHB

Storage wrapper:     CLSharedMemReleaseBuffer      VulkanExternalMemRelease
                     (already exists)              (already exists in buffer backend)

CPU→GPU path:        cl::Buffer ──[gl_to_cl]──►    VkBuffer ──[encodeBufferToTensor]──►
                     MNN cl::Buffer/Image           MNN VkImage (NC4HW4)

GPU→CPU path:        MNN cl::Buffer ──[cl_to_gl]──► MNN VkImage ──[encodeTensorToBuffer]──►
                     AHB cl::Buffer                 VkBuffer (AHB)

Format conversion:   GPU kernel (RGBA8/YUV420)     VulkanImageConverter (buffer↔image)
                     OpenCL compute shader          Existing encodeBufferToTensor pipeline

CPU memcpy:          NONE                          NONE

GPU copy:            Kernel dispatch                Compute dispatch (same cost)
```

### 6c. Why the Vulkan Image Backend Change Is Trivial

The OpenCL backend took ~80 lines of code to add full AHB support (`BufferConvertor.cpp`).
The Vulkan image backend change is even simpler because:

1. **Import is identical** — `VulkanBuffer::createExternalAHB()` already handles AHB→VkBuffer.
   It's compiled in BOTH image and buffer builds (`component/` is in both glob patterns).

2. **Format conversion already exists** — `VulkanImageConverter::encodeBufferToTensor()`
   already converts VkBuffer→VkImage. The image backend's `onCopyBuffer` host→gpu path
   already uses this pipeline. We just feed it the imported VkBuffer instead of `mHostBuffer`.

3. **Wrapper class exists** — `VulkanExternalMemRelease` (from buffer backend) handles the
   lifecycle. Same pattern as `CLSharedMemReleaseBuffer` in OpenCL.

4. **No CMake changes** — `VulkanBuffer.cpp` (with `createExternalAHB`) is in `component/`,
   compiled in all Vulkan builds. The change is only in `image/backend/VulkanBackend.cpp`.

### 6d. Implementation (completed)

The change is in a single file: `image/backend/VulkanBackend.cpp`. No new files,
no new CMake rules.

**What was added:**

1. **`VulkanExternalMemRelease` class** (mirrors buffer backend's same-named class):
   Wraps a `shared_ptr<VulkanBuffer>` + `int64_t handle`. Stored via
   `TensorUtils::setSharedMem()` to keep the imported VkBuffer alive.

2. **`onAcquire()` — AHB import path** (~30 lines):
   - Detects `MNN_MEMORY_AHARDWAREBUFFER` in `MTensor->buffer().flags`
   - Imports AHB as VkBuffer via `VulkanBuffer::createExternalAHB()` (Android)
     or `VulkanBuffer::createExternal()` (Linux dma-buf)
   - Stores as `VulkanExternalMemRelease` via `setSharedMem()`
   - Still creates `VulkanTensor` for compute shaders

3. **`onCopyBuffer()` — two new branches** (~65 lines):
   - **AHB → VkImage (input)**: Uses imported VkBuffer as source for
     `encodeBufferToTensor()`. Mirrors the host→gpu path but reads from the
     external VkBuffer instead of `mHostBuffer`.
   - **VkImage → AHB (output)**: Uses imported VkBuffer as destination for
     `encodeTensorToBuffer()`. Mirrors the gpu→host path but writes to the
     external VkBuffer instead of `mHostBuffer`.
   - Both paths use the same converter caching (`mConverters` map).

**Total: ~100 lines of new code.**

### 6e. Supported Formats

| Format | OpenCL | Vulkan (buffer) | Vulkan (image, proposed) |
|---|---|---|---|
| RGBA8 (AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM) | Yes | Yes | Yes |
| YUV420 (AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420) | Yes | No | No |
| RGBA32F (AHARDWAREBUFFER_FORMAT_R32G32B32A32_FLOAT) | N/A | Yes | Yes |

YUV420 is not supported in Vulkan because VulkanBuffer (storage buffer) works with raw bytes,
and the image backend's VkImage format is always R32G32B32A32_SFLOAT. A YUV→RGBA conversion
shader would be needed for full parity with OpenCL.

---

## 7. Gap Analysis: What's Missing

### 7a. ISP Extensions Are Disconnected

| Function | Implemented | Connected to pipeline? | Called by anyone? |
|---|---|---|---|
| `MNNVulkanQueryOptimalWorkgroup` | Hardcoded 8x8 | No | Nobody |
| `MNNVulkanSetSessionWorkgroup` | Stores in map | No | Nobody |
| `MNNVulkanSetWorkgroupPreset` | Empty | No | Nobody |
| `MNNVulkanHotSwapConstBuffer` | Stores bytes | No | Nobody |

### 7b. CI Builds Image Mode — Zero-Copy Not Compiled

| Build flag | Default (CI) | For zero-copy |
|---|---|---|
| `MNN_VULKAN_IMAGE` | `ON` (image) | Must be `OFF` (buffer) |
| `MNN_SEP_BUILD` | `ON` (separate .so) | Either works |
| `MNN_VULKAN` | `ON` | Must be `ON` |

The `buffer/backend/VulkanBackend.cpp` (containing the zero-copy `onAcquire`) is only compiled when `MNN_VULKAN_IMAGE=OFF`.

### 7c. Summary Table

| Capability | Where it lives | Build mode | Default? | Client-callable? | Functional? |
|---|---|---|---|---|---|
| Zero-copy AHB import | `libMNN_Vulkan.so` | Buffer only | No | Yes (`setDevicePtr`) | **Yes** |
| Zero-copy dma-buf import | `libMNN_Vulkan.so` | Buffer only | No | Yes (`setDevicePtr`) | **Yes** |
| ISP query workgroup | `libMNN_Vulkan.so` | Image or buffer | Yes | Yes (extern C) | **No** (stub) |
| ISP set workgroup | `libMNN_Vulkan.so` | Image or buffer | Yes | Yes (extern C) | **No** (stub) |
| ISP hot-swap const buf | `libMNN_Vulkan.so` | Image or buffer | Yes | Yes (extern C) | **No** (stub) |

---

## 6. Recommendations

### To make zero-copy work today (no MNN code changes):

1. **CI/build:** Add `-DMNN_VULKAN_IMAGE=OFF` to produce `libMNN_Vulkan.so` with buffer backend
2. **Deploy:** Ensure `libMNN_Vulkan.so` is in the same directory / APK as `libMNN.so`
3. **Client code:** Use `Tensor::setDevicePtr(ahb, MNN_MEMORY_AHARDWAREBUFFER)` — existing public API
4. **No ISP extension calls needed** — the zero-copy path is independent of them

### To make ISP extensions functional (future work):

- Wire `gSessionWorkgroups` / `gConstBuffers` globals into `VulkanRuntime`'s constant buffer management
- Add invalidation of Vulkan descriptor sets when const buffers change
- Add thread safety (`std::mutex` around the global maps)
- Expose the header as a public API (`include/MNN/VulkanISPExtensions.h`)
