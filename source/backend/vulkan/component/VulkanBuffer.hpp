//
//  VulkanBuffer.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBuffer_hpp
#define VulkanBuffer_hpp
#include "VulkanMemoryPool.hpp"
namespace MNN {
class VulkanBuffer : public NonCopyable {
public:
    VulkanBuffer(const VulkanMemoryPool& pool, bool separate, size_t size, const void* hostData = nullptr,
                 VkBufferUsageFlags usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VkSharingMode shared      = VK_SHARING_MODE_EXCLUSIVE,
                 VkFlags requirements_mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    // Imports an external memory handle (Linux V4L2 dma-buf fd) as a zero-copy
    // Vulkan buffer. The caller retains ownership of the fd; the VkDeviceMemory
    // is imported (not allocated) and is NOT freed on destruction.
    static std::shared_ptr<VulkanBuffer> createExternal(const VulkanMemoryPool& pool, int fd, size_t size,
                                                        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                        VkSharingMode shared = VK_SHARING_MODE_EXCLUSIVE);
#ifdef __ANDROID__
    // Imports an AHardwareBuffer as a zero-copy Vulkan buffer (Android only).
    static std::shared_ptr<VulkanBuffer> createExternalAHB(const VulkanMemoryPool& pool, AHardwareBuffer* ahb, size_t size, VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VkSharingMode shared = VK_SHARING_MODE_EXCLUSIVE);
#endif

    virtual ~VulkanBuffer();

    VkBuffer buffer() const {
        return mBuffer;
    }
    size_t size() const {
        return mSize;
    }
    void* map(int start = 0, int size = -1) const;
    void unmap() const;

    void flush(bool write, int start, int size) const;

    void release();
    bool external() const {
        return mExternal;
    }

private:
    VulkanBuffer(const VulkanMemoryPool& pool, int fd, size_t size, VkBufferUsageFlags usage, VkSharingMode shared);
#ifdef __ANDROID__
    VulkanBuffer(const VulkanMemoryPool& pool, AHardwareBuffer* ahb, size_t size, VkBufferUsageFlags usage, VkSharingMode shared);
#endif

    const VulkanMemoryPool& mPool;
    MemChunk mMemory;
    VkBuffer mBuffer;
    size_t mSize;
    VkBufferUsageFlags mUsage;
    bool mReleased = false;
    VkSharingMode mShared;
    bool mExternal = false;
};
} // namespace MNN

#endif /* VulkanBuffer_hpp */
