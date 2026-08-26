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
    // HOST_COHERENT is REQUIRED: MNN maps host-visible buffers and reads them
    // without calling vkInvalidateMappedMemoryRanges (VulkanBuffer::flush is
    // a no-op). A HOST_CACHED non-coherent type satisfies HOST_VISIBLE alone,
    // and on some Adreno drivers the CPU then sees stale data in a periodic
    // pattern (only the first 96B of every 384B window updated).
    VulkanBuffer(const VulkanMemoryPool& pool, bool separate, size_t size, const void* hostData = nullptr,
                 VkBufferUsageFlags usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VkSharingMode shared      = VK_SHARING_MODE_EXCLUSIVE,
                 VkFlags requirements_mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

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

private:
    const VulkanMemoryPool& mPool;
    MemChunk mMemory;
    VkBuffer mBuffer;
    size_t mSize;
    VkBufferUsageFlags mUsage;
    bool mReleased = false;
    VkSharingMode mShared;
};
} // namespace MNN

#endif /* VulkanBuffer_hpp */
