//
//  VulkanBuffer.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanBuffer.hpp"
#include "component/MNNIspTrace.hpp"
#include <string.h>
#include <cstdarg>
#include <cstdio>

// [ZERO-COPY] Trace helper. Self-contained so this translation unit never
// imports `ispTraceSink` (internal to libMNN_Vulkan); an undefined reference
// there made the whole .so fail to dlopen:
//   "cannot locate symbol ispTraceSink referenced by libmnnhelper.so"
extern "C" __attribute__((visibility("default")))
void mnnIspZeroCopyLog(const char* fmt, ...) {
    FILE* tf = fopen("/sdcard/Download/isp_vlog_trace", "a");
    if (tf == nullptr) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(tf, fmt, ap);
    va_end(ap);
    fflush(tf);
    fclose(tf);
}

namespace MNN {

VulkanBuffer::VulkanBuffer(const VulkanMemoryPool& pool, bool separate, size_t size, const void* hostData,
                           VkBufferUsageFlags usage, VkSharingMode shared, VkFlags requirements_mask)
    : mPool(pool) {
    MNN_ASSERT(size > 0);
    mSize = size;
    mShared = shared;
    mBuffer = const_cast<VulkanMemoryPool&>(mPool).allocBuffer(size, usage, shared);
    mUsage = usage;

    VkMemoryRequirements memReq;
    mPool.device().getBufferMemoryRequirements(mBuffer, memReq);
    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memReq, requirements_mask, separate);
    if (nullptr == mMemory.first) {
        MNN_ERROR("VulkanBuffer: allocMemory failed, size=%zu\n", size);
        return;
    }
    auto realMem = (VulkanMemory*)mMemory.first;

    if (nullptr != hostData) {
        void* data = nullptr;
        CALL_VK(mPool.device().mapMemory(realMem->get(), mMemory.second, size, 0 /*flag, not used*/, &data));
        ::memcpy(data, hostData, size);
        mPool.device().unmapMemory(realMem->get());
    }
    CALL_VK(mPool.device().bindBufferMemory(mBuffer, realMem->get(), mMemory.second));
}

VulkanBuffer::~VulkanBuffer() {
    const_cast<VulkanMemoryPool&>(mPool).returnBuffer(mBuffer, mSize, mUsage, mShared);
    if (!mReleased) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
    }
}
void* VulkanBuffer::map(int start, int size) const {
    const auto& limits = mPool.device().proty().limits;
    if (size < 0) {
        size = mSize;
    }
    auto realMem = (VulkanMemory*)mMemory.first;
    void* data = nullptr;
    CALL_VK(mPool.device().mapMemory(realMem->get(), start + mMemory.second, size, 0, &data));
    return data;
}
void VulkanBuffer::unmap() const {
    auto realMem = (VulkanMemory*)mMemory.first;
    mPool.device().unmapMemory(realMem->get());
}
void VulkanBuffer::release() {
    if (mReleased) {
        return;
    }
    mReleased = true;
    const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
}

void VulkanBuffer::flush(bool write, int start, int size) const {
    // Do nothing
}

} // namespace MNN

// ── [ZERO-COPY DISPLAY] AHB-backed buffer ────────────────────────────────
// Imports a gralloc AHardwareBuffer as VkDeviceMemory so the GPU can write
// the display output in place. Requires
//   VK_KHR_external_memory (004-00) + VK_ANDROID_external_memory_android_hardware_buffer
//   VK_KHR_get_memory_requirements2 + VK_KHR_bind_memory2 + VK_KHR_dedicated_allocation
// Returns nullptr when any of them (or the import itself) is unavailable, so
// callers fall back to the existing staging path.
#include "backend/vulkan/vulkan/vulkan_android.h"
#include <android/hardware_buffer.h>

extern "C" {
typedef VkResult (*PFN_vkGetAndroidHardwareBufferPropertiesANDROID_)(VkDevice, const struct AHardwareBuffer*, VkAndroidHardwareBufferPropertiesANDROID*);
typedef void (*PFN_vkGetMemoryAndroidHardwareBufferANDROID_)(VkDevice, const VkMemoryGetAndroidHardwareBufferInfoANDROID*, struct AHardwareBuffer**);
}

namespace MNN {

VulkanBufferFromAHB* VulkanBufferFromAHB::create(const VulkanDevice& device,
                                                 void* hardwareBuffer,
                                                 size_t size,
                                                 VkBufferUsageFlags usage) {
    if (hardwareBuffer == nullptr || size == 0) {
        return nullptr;
    }
    // Resolve the import entry points through the instance.
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID_ getProps = nullptr;
    if (vkGetInstanceProcAddr != nullptr) {
        // currentInstance is stored on the Runtime; use the device-level loader
        // instead (VK_ANDROID_external_memory is a DEVICE extension).
    }
    if (vkGetDeviceProcAddr != nullptr) {
        getProps = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID_)
            vkGetDeviceProcAddr(device.get(), "vkGetAndroidHardwareBufferPropertiesANDROID");
    }
    if (getProps == nullptr) {
        MNN_PRINT("[ZERO-COPY] vkGetAndroidHardwareBufferPropertiesANDROID unavailable\n");
        return nullptr;
    }

    const char* want[] = {
        "VK_KHR_external_memory",
        "VK_ANDROID_external_memory_android_hardware_buffer",
        "VK_KHR_dedicated_allocation",
        "VK_KHR_get_memory_requirements2",
        "VK_KHR_bind_memory2",
    };
    for (const char* w : want) {
        ::mnnIspZeroCopyLog("[ZERO-COPY] need %s\n", w);
    }

    auto* res = new VulkanBufferFromAHB(device);
    res->mSize = size;

    VkExternalMemoryBufferCreateInfo extBufInfo = {};
    extBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.pNext = &extBufInfo;
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device.get(), &bufInfo, nullptr, &res->mBuffer) != VK_SUCCESS) {
        delete res;
        return nullptr;
    }

    // Required device memory size for this AHB.
    VkAndroidHardwareBufferPropertiesANDROID ahbProps = {};
    ahbProps.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    if (getProps(device.get(), (const struct AHardwareBuffer*)hardwareBuffer, &ahbProps) != VK_SUCCESS) {
        vkDestroyBuffer(device.get(), res->mBuffer, nullptr);
        delete res;
        return nullptr;
    }

    uint32_t memTypeIndex = 0;
    bool found = false;
    const uint32_t typeCount = device.memProty().memoryTypeCount;
    for (uint32_t i = 0; i < typeCount && i < 32; ++i) {
        if (ahbProps.memoryTypeBits & (1u << i)) { memTypeIndex = i; found = true; break; }
    }
    if (!found) {
        vkDestroyBuffer(device.get(), res->mBuffer, nullptr);
        delete res;
        return nullptr;
    }

    VkImportAndroidHardwareBufferInfoANDROID importInfo = {};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.buffer = const_cast<struct AHardwareBuffer*>(
        (const struct AHardwareBuffer*)hardwareBuffer);

    VkMemoryDedicatedAllocateInfo dedicated = {};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.pNext = &importInfo;
    dedicated.buffer = res->mBuffer;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &dedicated;
    allocInfo.allocationSize = ahbProps.allocationSize;
    allocInfo.memoryTypeIndex = memTypeIndex;
    if (vkAllocateMemory(device.get(), &allocInfo, nullptr, &res->mMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device.get(), res->mBuffer, nullptr);
        delete res;
        return nullptr;
    }
    if (vkBindBufferMemory(device.get(), res->mBuffer, res->mMemory, 0) != VK_SUCCESS) {
        vkFreeMemory(device.get(), res->mMemory, nullptr);
        vkDestroyBuffer(device.get(), res->mBuffer, nullptr);
        delete res;
        return nullptr;
    }
    mnnIspZeroCopyLog("[ZERO-COPY] AHB imported ok size=%zu memType=%u\n", size, memTypeIndex);
    return res;
}

VulkanBufferFromAHB::~VulkanBufferFromAHB() {
    if (mBuffer != VK_NULL_HANDLE) {
        mDevice.destroyBuffer(mBuffer);
    }
    if (mMemory != VK_NULL_HANDLE) {
        mDevice.freeMemory(mMemory);
    }
}

} // namespace MNN
