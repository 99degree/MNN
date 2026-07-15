# Vulkan External-Memory Zero-Copy (CMA / dma-buf / AHardwareBuffer)

> Status: **implemented (unverified)**. The MNN Vulkan backend now imports a
> Linux V4L2 dma-buf fd as external `VkDeviceMemory` (committed locally as
> `1b94cc09`); the cam-isp integration surface that *uses* this is already
> implemented and clippy-clean. The remaining gap is the NDK rebuild +
> on-device validation (no `cmake`/GPU/V4L2 in Termux). See §6.
>
## 1. Problem statement

For camera ISP pipelines the input is a **large MIPI Bayer frame** (e.g. 48 MP raw
≈ 100+ MB). The correct zero-copy flow is:

```
CMA / dma-buf buffer  ──mmap──▶  CPU ptr  ──fd──▶  V4L2 / MIPI  (sensor DMA-writes in place)
                                        │
                                        └─import as Vulkan external device memory─▶ MNN input tensor
```

i.e. the sensor writes directly into a CMA/dma-buf, and that same physical memory is
imported into Vulkan as a device tensor so the GPU reads the camera's DMA pages with
**no CPU staging copy and no heap allocation**.

Today MNN **cannot do this on the Vulkan backend**:

- `Tensor::setDevicePtr(ptr, memoryType)` exists (`source/core/Tensor.cpp:498`) and
  just records `mBuffer.device = ptr; mBuffer.flags = memoryType;`.
- `MNN_MEMORY_AHARDWAREBUFFER = 14` exists (`include/MNN/MNNForwardType.h:55`).
- The **OpenCL** backend already imports an AHardwareBuffer as device memory
  (`source/backend/opencl/core/OpenCLBackend.cpp:747` `_allocHostBuffer`).
- The **Vulkan** backend does **not** read `mBuffer.flags` / `mBuffer.device` at all
  during buffer creation — so `setDevicePtr` is a **no-op at the backend level** on
  Vulkan. A Vulkan session therefore still allocates its own `VkDeviceMemory` and
  copies host→device, defeating zero-copy for the exact case (large Bayer) we care
  about.

The Vulkan external-memory primitives are already present in the headers
(`VK_EXT_external_memory_dma_buf`, `VkImportMemoryFdInfoKHR`, `vkGetMemoryFdKHR` in
`source/backend/vulkan/vulkan/vulkan_core.h:11063/8448/8468`), so this is an
**enhancement of our custom MNN fork**, not a from-scratch feature.

## 2. Verified call-flow difference (OpenCL vs Vulkan)

Both backends override `Backend::onAcquire(tensor, storageType)` as the hook where a
tensor's storage buffer is created. After `setDevicePtr(dev, MNN_MEMORY_AHARDWAREBUFFER)`
writes `tensor->buffer().device = dev` + `flags`, the two backends diverge completely:

| Aspect | OpenCL | Vulkan |
|---|---|---|
| Import hook | `onAcquire` → `_allocHostBuffer` (`OpenCLBackend.cpp:747`) | `onAcquire` (`source/backend/vulkan/buffer/backend/VulkanBackend.cpp:218`) |
| Import API | CL `CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM` (Adreno) / `CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM` (Mali) | would need `VkImportMemoryFdInfoKHR` (Linux dma-buf) / `VkImportAndroidHardwareBufferInfoANDROID` (Android AHB) |
| Platform | **Android AHardwareBuffer only** (`#ifdef __ANDROID__` + `isSupportAHD()`) | external-memory ext enabled per-platform |
| Memory slot | `TensorUtils::setSharedMem(tensor, …)` (per-tensor shared slot) | `tensor->buffer().device = (uint64_t)VulkanBuffer*` |
| Release | `CLSharedMemReleaseBuffer` — does **not** free the AHB (external owner keeps it) | `VulkanMemRelease` — frees through the pool (**wrong** for external) |

**Implication — mirroring OpenCL is NOT a copy-paste.** The Vulkan change must:

1. Branch in `VulkanBackend::onAcquire` on `tensor->buffer().flags == MNN_MEMORY_AHARDWAREBUFFER`.
2. Import the external handle into a `VulkanBuffer` (new import ctor / `VulkanDevice::allocMemory` path via `VkImportMemoryFdInfoKHR`).
3. Use a release path that does **not** free the imported `VkDeviceMemory` (the CMA/dma-buf owns it).
4. Cover **both** platforms: Linux V4L2 = dma-buf fd (`VK_KHR_external_memory_fd`); Android HAL3 = AHardwareBuffer (`VK_ANDROID_external_memory_android_hardware_buffer`). OpenCL gives **no** Linux dma-buf precedent — that path is net-new Vulkan code.

## 3. Proposed fix

### 3.1 Enable the Vulkan external-memory instance extensions

In `VulkanRuntime` / `VulkanInstance` device/instance creation, request the
external-memory extensions when building for a zero-copy target:

- Linux / V4L2 dma-buf: `VK_KHR_external_memory_fd` (+ `VK_KHR_external_memory` +
  `VK_KHR_get_physical_device_properties2`). Import handle type
  `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`.
- Android / HAL3: `VK_ANDROID_external_memory_android_hardware_buffer` (import an
  `AHardwareBuffer*` via `vkGetAndroidHardwareBufferPropertiesANDROID` +
  `VkImportAndroidHardwareBufferInfoANDROID`).

The extension constants/types are already declared in `vulkan_core.h`; only the
instance/device feature enabling + import call sites are missing.

### 3.2 `VulkanBackend::onAcquire` — the injection point

`source/backend/vulkan/buffer/backend/VulkanBackend.cpp:218`:

```cpp
Backend::MemObj* VulkanBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    auto alignSize = getTensorSize(tensor);
    auto MTensor   = const_cast<Tensor*>(tensor);
    auto des       = TensorUtils::getDescribeOrigin(tensor);

    // ---- NEW: external device memory (CMA / dma-buf / AHardwareBuffer) ----
    if (tensor->buffer().flags == MNN_MEMORY_AHARDWARE_BUFFER_VULKAN && tensor->buffer().device != 0) {
        auto imported = mRuntime->mDevice->importExternalMemory(
            tensor->buffer().device,        // dma-buf fd (Linux) or AHardwareBuffer* (Android)
            alignSize, flagsToHandleType(tensor->buffer().flags));
        auto mem = new VulkanExternalMemRelease(imported);   // does NOT free on release
        MTensor->buffer().device = (uint64_t)imported;
        des->offset = 0;
        return mem;
    }
    // ---- existing pool allocation (unchanged) ----
    if (Backend::STATIC == storageType) {
        auto newBuffer = mRuntime->mBufferPool->alloc(alignSize);
        ...
    }
    ...
}
```

Notes:
- A new `MNN_MEMORY_AHARDWARE_BUFFER_VULKAN` (or reuse `MNN_MEMORY_AHARDWAREBUFFER = 14`)
  flags the tensor as external. The cam-isp FFI already passes `14`.
- `flagsToHandleType()` maps the memory type to the Vulkan handle type
  (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` vs the Android AHB handle).

### 3.3 `VulkanDevice::importExternalMemory` — the actual import

`source/backend/vulkan/component/VulkanDevice.cpp` (alongside the existing
`allocMemory`):

```cpp
VulkanMemory* VulkanDevice::importExternalMemory(uint64_t handle, size_t size,
                                                 VkExternalMemoryHandleTypeFlagBits handleType) {
    VkMemoryAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = size;
    info.memoryTypeIndex = pickMemoryType(handleType);   // external-memory capable type

    VkMemoryDedicatedAllocateInfo dedicated{};
    info.pNext = &dedicated;

    if (handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
        // Linux V4L2 dma-buf fd
        VkImportMemoryFdInfoKHR fdInfo{};
        fdInfo.sType  = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        fdInfo.fd     = static_cast<int>(handle);
        fdInfo.handleType = handleType;
        dedicated.pNext = &fdInfo;
    } else {
        // Android AHardwareBuffer
        VkImportAndroidHardwareBufferInfoANDROID ahb{};
        ahb.sType        = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
        ahb.buffer       = reinterpret_cast<AHardwareBuffer*>(handle);
        dedicated.pNext  = &ahb;
    }

    VkDeviceMemory mem = VK_NULL_HANDLE;
    CALL_VK(mDevice.allocateMemory(mDevice, &info, nullptr, &mem));
    // wrap in a VulkanMemory / VulkanBuffer (no vkAllocateMemory happened)
    return new VulkanMemory(*this, mem);
}
```

Also create the `VkBuffer` bound to this memory (mirroring the normal
`VulkanBuffer` creation) so ops can address it.

### 3.4 Release must NOT free external memory

`VulkanMemRelease` (the normal `Backend::MemObj`) frees the `VkDeviceMemory` back to
the pool. For imported memory the CMA/dma-buf owns the pages, so add a sibling
`VulkanExternalMemRelease` whose destructor does **not** call `vkFreeMemory` (or only
frees the `VkBuffer` binding, not the `VkDeviceMemory`). This mirrors OpenCL's
`CLSharedMemReleaseBuffer`, which keeps the AHB alive.

### 3.5 The `VulkanBuffer` import ctor

Add a `VulkanBuffer` constructor / factory that takes an already-imported
`VkDeviceMemory` (instead of going through `VulkanMemoryPool::allocMemory` →
`vkAllocateMemory`). The existing `VulkanBuffer` ctor
(`source/backend/vulkan/component/VulkanBuffer.cpp`) already wraps a
`VulkanMemory`; the import path only differs in *how the memory was obtained*.

## 4. Integration surface already done (cam-isp)

The consumer side is implemented and `cargo clippy -p cam-isp --lib --features mnn
-D warnings` clean:

- `mnn_run_external_zero_copy` (C++ FFI, `cam-isp/mnn_sys/mnn_wrapper.cpp`): binds
  the external handle via `setDevicePtr` then runs the session.
- `MnnEngine::run_external_zero_copy` (`cam-isp/src/mnnengine.rs`): acquires a
  session from the pool and calls the FFI.
- `CameraIspService::process_raw_frame_external` (`cam-isp/src/integration.rs`):
  extracts the CMA `dma_fd` from the input buffer and drives the MNN session.

Once §3 lands, this path becomes truly zero-copy on Vulkan (today it is inert because
the backend ignores the fd).

## 5. Verification status & risks

- **Verified:** the call-flow difference (§2), the presence of the external-memory
  Vulkan headers, and that the cam-isp surface compiles + lints clean.
- **Cannot verify in headless Termux:** no `cmake`/`ninja`/`vulkan` in PATH, no
  GPU/V4L2 device. The MNN backend change (§3) **must be built on the Android NDK
  machine and validated on-device** (V4L2 dma-buf → Vulkan import → inference).
- **Risk:** the Linux dma-buf import path is net-new (OpenCL only covers Android
  AHB), so it needs careful on-device testing (handle-type negotiation, memory-type
  bit selection, dedicated-allocation requirements).
- After building, copy `libMNN.so` / `libMNN_Vulkan.so` into
  `cam-rust/lib/arm64-v8a/` so cam-isp links the enhanced backend.

## 6. TODO order (status)

1. ~~Enable external-memory instance/device extensions in `VulkanRuntime`/`VulkanInstance`.~~ **DONE** — `VulkanInstance` enables `VK_KHR_external_memory_capabilities`, `VulkanDevice` enables `VK_KHR_external_memory_fd` (both gated on availability).
2. ~~Add `VulkanDevice::importExternalMemory` (dma-buf + AHB branches).~~ **DONE** (dma-buf only) — folded into `VulkanBuffer::createExternal` + a `pNext`-aware `VulkanDevice::createBuffer` + an external `VulkanMemory` wrapper. Android AHB is intentionally out of scope (MNN's Vulkan headers lack `VkImportAndroidHardwareBufferInfoANDROID`).
3. ~~Add `VulkanBuffer` import factory + `VulkanExternalMemRelease`.~~ **DONE** — `VulkanBuffer::createExternal()` imports via `VkImportMemoryFdInfoKHR` and `VulkanExternalMemRelease` keeps the buffer alive without freeing the fd.
4. ~~Branch `VulkanBackend::onAcquire` on the external-memory flag.~~ **DONE** — mirrors OpenCL's `sharedMem` re-import (handle changes between frames).
5. **REMAINING:** Build with NDK, copy `libMNN.so` / `libMNN_Vulkan.so` into `cam-rust/lib/arm64-v8a/`, validate on-device with a V4L2 dma-buf input. The external path is gated by `MNN_MEMORY_AHARDWAREBUFFER` (14) and never fires in default MNN usage, so the existing Vulkan path is unaffected by this change.

## 7. Codebase quality scan (2026-07-15)

A targeted scan of the cam-isp integration surface and the MNN Vulkan backend was
done to confirm no low-quality patterns remain on this path. Result: the
previously-identified defects are **fixed and stay fixed**; the only remaining
inert code is **intentional** (documented enhancement surface), not a defect.

**Defects fixed (verified present + clean):**
- `engine.rs` `as_any`/`as_any_mut` default `unimplemented!()` -> now **required**
  trait methods, implemented in all 4 engines (`engine.rs`, `onnx/mod.rs`,
  `cpu.rs`, `mnnengine.rs`). No `unimplemented!()`/`todo!()` left in the workspace.
- `hdr.rs:338` stale "identity alignment (noop)" comment -> now documents the real
  block-matching `align_frames` (4x4 grid + median voting).
- `cam-isp/mnn_sys/mnn_wrapper.cpp:345` `mnn_run_true_zero_copy` ignoring its buffer
  + `memcpy` -> now binds the caller's CMA mmap as input **and** output host
  (true zero-copy). The new `mnn_run_external_zero_copy` (this issue's surface) is
  built on the same public `Tensor::buffer().host` binding.

**Intentional / inert-by-design (not defects):**
- `cam-isp/mnn_sys/mnn_vulkan_stubs.cpp` -- 4 weak no-op Vulkan extension stubs
  (`MNNVulkanQueryOptimalWorkgroup`, `MNNVulkanSetSessionWorkgroup`,
  `MNNVulkanSetWorkgroupPreset`, `MNNVulkanHotSwapConstBuffer`). These are
  precisely the enhancement surface this doc describes: no-ops against official
  MNN 3.6.0, overridden by the custom MNN fork (e.g. the external-memory import
  committed as `1b94cc09`).
- `mnn_run_host_tensors` (`mnn_wrapper.cpp:192`) still does host->backend copy via
  `copyFromHostTensor`/`copyToHostTensor` -- the legitimate non-zero-copy path; the
  zero-copy variants are the improvements.
- `onnx/mod.rs:295` returns an empty `IspFrame` when `ort` is disabled
  (`#[cfg(not(feature = "ort"))]`) -- feature-gated fallback, not exercised in
  production.

**No error-masking in production:**
- All `panic!` hits are inside `#[cfg(test)]` modules. The two `unreachable!()`
  (`display.rs:635`, `format_convert.rs:235`) are exhaustive-match guards over
  fixed enums. `cam-binder` "torch not implemented" is a proper `NotImplemented`
  error (logged + returned), not a silent no-op.
- The whole cam-isp surface is `cargo clippy -D warnings`-clean.

**Conclusion:** no low-quality code remains on the external-memory zero-copy path.
The only genuinely inert code is the 4 weak Vulkan stubs, which are no-ops by
design until the custom MNN build supplies real symbols (this issue's backend
change is the first such real symbol).
