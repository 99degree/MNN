//
//  VulkanBuffer.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanBuffer.hpp"
#include <string.h>
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
    //        FUNC_PRINT(mMemory->type());
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

VulkanBuffer::VulkanBuffer(const VulkanMemoryPool& pool, int fd, size_t size, VkBufferUsageFlags usage,
                           VkSharingMode shared)
    : mPool(pool) {
    MNN_ASSERT(size > 0);
    MNN_ASSERT(fd >= 0);
    mSize     = size;
    mShared   = shared;
    mUsage    = usage;
    mExternal = true;

    VkExternalMemoryBufferCreateInfo extBufferInfo{};
    extBufferInfo.sType      = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufferInfo.pNext      = nullptr;
    extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    mBuffer = VK_NULL_HANDLE;
    CALL_VK(mPool.device().createBuffer(mBuffer, size, usage, shared, &extBufferInfo));

    VkMemoryRequirements memReq{};
    mPool.device().getBufferMemoryRequirements(mBuffer, memReq);

    VkImportMemoryFdInfoKHR importFd{};
    importFd.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    importFd.pNext      = nullptr;
    importFd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    importFd.fd         = fd;

    // Choose a memory type compatible with the dma-buf import. Iterate over the
    // buffer's supported types, preferring device-local, and import on the
    // first one that accepts the fd (an import allocate is cheap).
    VkDeviceMemory devMem = VK_NULL_HANDLE;
    uint32_t typeIndex = 0;
    const auto& memTypes  = mPool.device().memProty().memoryTypes;
    uint32_t typeCount    = mPool.device().memProty().memoryTypeCount;
    auto tryImport = [&](uint32_t i) -> bool {
        if ((memReq.memoryTypeBits & (1u << i)) == 0) {
            return false;
        }
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext           = &importFd;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = i;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (VK_SUCCESS != mPool.device().allocMemory(mem, allocInfo)) {
            return false;
        }
        if (VK_SUCCESS != mPool.device().bindBufferMemory(mBuffer, mem, 0)) {
            mPool.device().freeMemory(mem);
            return false;
        }
        devMem = mem;
        typeIndex = i;
        return true;
    };
    bool imported = false;
    for (uint32_t i = 0; i < typeCount; i++) {
        if ((memTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && tryImport(i)) {
            imported = true;
            break;
        }
    }
    if (!imported) {
        for (uint32_t i = 0; i < typeCount; i++) {
            if (tryImport(i)) {
                imported = true;
                break;
            }
        }
    }
    if (!imported) {
        mPool.device().destroyBuffer(mBuffer);
        mBuffer = VK_NULL_HANDLE;
        MNN_ERROR("VulkanBuffer: failed to import dma-buf fd %d as Vulkan memory\n", fd);
        return;
    }
    mMemory = MemChunk(new VulkanMemory(mPool.device(), devMem, typeIndex, memReq.size), 0);
}

std::shared_ptr<VulkanBuffer> VulkanBuffer::createExternal(const VulkanMemoryPool& pool, int fd, size_t size,
                                                          VkBufferUsageFlags usage, VkSharingMode shared) {
    auto buffer = std::shared_ptr<VulkanBuffer>(new VulkanBuffer(pool, fd, size, usage, shared));
    if (VK_NULL_HANDLE == buffer->mBuffer) {
        return nullptr;
    }
    return buffer;
}

} // namespace MNN
