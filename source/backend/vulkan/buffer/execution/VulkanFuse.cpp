//
//  VulkanFuse.cpp
//  MNN
//
//  Created by MNN on 2023/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <cstring>
#include <cstdint>
// Explicitly include buffer backend's VulkanBackend.hpp (has getPipelineFactory/getBuffer)
// to avoid image/backend shadowing it.
#include "../backend/VulkanBackend.hpp"
#include "VulkanBasicExecution.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanFuse.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {

VulkanFuse::VulkanFuse(const Extra* extra, Backend* bn, int inputSize, int outputSize) : VulkanBasicExecution(bn) {
    auto vkBn = static_cast<VulkanBackend*>(bn);
    auto factory = vkBn->getPipelineFactory();
    mOutputBinding.resize(outputSize);
    mInputBinding.resize(inputSize);
    mGroupSize.resize(3);
    mGlobalSize.resize(3);
    // Find shader
    const uint8_t* data = nullptr;
    size_t dataSize = 0;
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "spirv") {
            data = (uint8_t*)attr->tensor()->int8s()->data();
            dataSize = attr->tensor()->int8s()->size();
            break;
        }
    }

    // Helper lambda to read ints from either list or tensor attribute
    auto readInts = [](const Attribute* attr, int* out, int count) -> bool {
        if (attr->list() && attr->list()->i()) {
            auto data = attr->list()->i();
            for (int j = 0; j < count && j < data->size(); j++) {
                out[j] = data->data()[j];
            }
            return true;
        }
        if (attr->tensor()) {
            if (attr->tensor()->int32s()) {
                auto data = attr->tensor()->int32s();
                for (int j = 0; j < count && j < data->size(); j++) {
                    out[j] = data->data()[j];
                }
                return true;
            }
            if (attr->tensor()->int8s()) {
                auto data = attr->tensor()->int8s();
                for (int j = 0; j < count && j < data->size(); j++) {
                    out[j] = data->data()[j];
                }
                return true;
            }
        }
        return false;
    };
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "global_size") {
            readInts(attr, mGlobalSize.data(), 3);
            mNeedAutoTuning = true;
            break;
        }
    }
    // Read preferred workgroup size from group_size attribute.
    // This is the LOCAL_SIZE the shader uses (e.g. 16×16).
    // optimized_dispatch uses this to compute dispatch group count.
    mPreferredLocalSize = {16, 16, 1};
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "group_size") {
            readInts(attr, mPreferredLocalSize.data(), 3);
            mNeedAutoTuning = false;
            break;
        }
    }
    // Optimization: Check for minimal dispatch mode
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "optimized_dispatch") {
            mOptimizedDispatch = attr->b();
            break;
        }
    }
    // Early-Z: Skip workgroups entirely outside valid image bounds.
    // Attr "early_z" = true, "valid_bounds" = {x0, y0, x1, y1} in pixels.
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "early_z") {
            mEarlyZ = attr->b();
        }
        if (attr->key()->str() == "valid_bounds") {
            if (attr->list() && attr->list()->i()) {
                auto data = attr->list()->i();
                for (int j = 0; j < 4 && j < data->size(); j++) {
                    mValidBounds.push_back(data->data()[j]);
                }
            }
        }
    }
    // FP16 const packing: when fp16_consts=true, pack const buffers as FP16
    // to halve GPU→shader bandwidth. Shader reads via unpackHalf2x16.
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "fp16_consts") {
            mFp16Consts = attr->b();
            break;
        }
    }
    // [VulkanFuse] mOptimizedDispatch set
    // ... rest of constructor ...
    std::vector<VkDescriptorType> types;
    int maxIndex = -1;
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "input") {
            auto list = attr->list()->i()->data();
            int binding = (int)list[1];
            maxIndex = ALIMAX(maxIndex, binding);
        } else if (attr->key()->str() == "const") {
            maxIndex = ALIMAX(maxIndex, (int)attr->i());
        }
    }
    types.resize(maxIndex+1);
    std::vector<std::tuple<int, void*, size_t>> constStoragePtrs;
    std::vector<std::tuple<int, void*, size_t>> constUniformPtrs;
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "input") {
            auto list = attr->list()->i()->data();
            if (list[1] >= 0) {
                if (0 == list[0]) {
                    mInputBinding[attr->i()] = list[1];
                } else {
                    mOutputBinding[attr->i()] = list[1];
                }
            }
            {
                int binding = (int)list[1];
                if (binding >= 0 && binding < types.size()) {
                    if (attr->b()) {
                        types[binding] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    } else {
                        types[binding] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    }
                }
            }
            continue;
        }
        if (attr->key()->str() == "const") {
            auto b = attr->tensor();
            void* result = nullptr;
            size_t bufferSize = 0;
            switch (b->dataType()) {
                case DataType_DT_FLOAT:
                    result = (void*)b->float32s()->Data();
                    bufferSize = b->float32s()->size() * sizeof(float);
                    break;
                case DataType_DT_INT32:
                    result = (void*)b->int32s()->Data();
                    bufferSize = b->int32s()->size() * sizeof(float);
                    break;
                default:
                    MNN_ASSERT(false);
                    break;
            }
            // FP16 const packing: when fp16_consts=true, convert f32→f16
            // to halve const buffer bandwidth. Shader reads via unpackHalf2x16.
            if (mFp16Consts && b->dataType() == DataType_DT_FLOAT) {
                int count = b->float32s()->size();
                // Pack as FP16: each float → half (2 bytes)
                // Store as array of uint32 where each holds 2 packed halfs
                std::vector<uint16_t> fp16(count);
                for (int j = 0; j < count; j++) {
                    float v = b->float32s()->data()[j];
                    // Simple round-to-nearest FP16
                    uint32_t f = *((uint32_t*)&v);
                    uint32_t sign = (f >> 16) & 0x8000;
                    int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
                    uint32_t mantissa = (f >> 13) & 0x3FF;
                    if (exponent <= 0) { fp16[j] = (uint16_t)sign; }
                    else if (exponent >= 31) { fp16[j] = (uint16_t)(sign | 0x7BFF); }
                    else { fp16[j] = (uint16_t)(sign | (exponent << 10) | mantissa); }
                }
                result = fp16.data();
                bufferSize = count * sizeof(uint16_t);
                // Store fp16 data to keep alive during merge
                mFp16DataStorage = std::vector<uint8_t>((uint8_t*)fp16.data(), (uint8_t*)fp16.data() + bufferSize);
                result = mFp16DataStorage.data();
            }
            if (attr->b()) {
                types[attr->i()] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                constUniformPtrs.emplace_back(std::make_tuple(attr->i(), result, bufferSize));
            } else {
                types[attr->i()] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                constStoragePtrs.emplace_back(std::make_tuple(attr->i(), result, bufferSize));
            }
            continue;
        }
    }
    auto alignSize = vkBn->device().proty().limits.minMemoryMapAlignment;
    size_t offset = 0;
    std::shared_ptr<VulkanCommandPool::Buffer> cmdbuffer( vkBn->getPool().allocBuffer());
    cmdbuffer->begin(0);
    auto merge = [&](const std::vector<std::tuple<int, void*, size_t>>& constPtrs, VkDescriptorType type) {
        if (constPtrs.empty()) {
            return std::make_tuple(std::vector<std::tuple<int, size_t, size_t>>{}, std::shared_ptr<VulkanBuffer>(nullptr), std::shared_ptr<VulkanBuffer>(nullptr));
        }
        std::vector<std::tuple<int, size_t, size_t>> mConstOffset;
        for (auto& constAttr : constPtrs) {
            // Optimization: Power-of-2 alignment for faster GPU access
            auto size = UP_DIV(std::get<2>(constAttr), alignSize) * alignSize;
            size = std::max<size_t>(size, 256); // Minimum 256-byte alignment
            mConstOffset.emplace_back(std::make_tuple(std::get<0>(constAttr), size, offset));
            offset += size;
        }
        std::shared_ptr<VulkanBuffer> hostBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, offset, nullptr, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
        auto ptr = (uint8_t*)hostBuffer->map();
        for (int i=0; i<constPtrs.size(); ++i) {
            ::memcpy(ptr + std::get<2>(mConstOffset[i]), std::get<1>(constPtrs[i]), std::get<2>(constPtrs[i]));
        }
        hostBuffer->unmap();
        VkBufferUsageFlags bufferUsage = (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) 
            ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT 
            : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        std::shared_ptr<VulkanBuffer> vkBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, offset, nullptr, 
            bufferUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
        VkBufferCopy bufferCopy;
        bufferCopy.size = offset;
        bufferCopy.dstOffset = 0;
        bufferCopy.srcOffset = 0;
        vkCmdCopyBuffer(cmdbuffer->get(), hostBuffer->buffer(), vkBuffer->buffer(),
                        1, &bufferCopy);
        return std::make_tuple(mConstOffset, vkBuffer, hostBuffer);
    };
    mConstStorageOffset.clear();
    mConstUniformOffset.clear();
    auto uniforms = merge(constUniformPtrs, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    mConstUniformOffset = std::get<0>(uniforms);
    mConstUniformBuffer = std::get<1>(uniforms);
    auto storages = merge(constStoragePtrs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    mConstStorageOffset = std::get<0>(storages);
    mConstStorageBuffer = std::get<1>(storages);
    cmdbuffer->end();
    auto fence = vkBn->getPool().submit(cmdbuffer->get());

    mPipeline = factory->createComputePipeline(data, dataSize, types, std::vector<uint32_t>{});
    mDescriptorSet = mPipeline->createSet();
    fence->wait();
}

VulkanFuse::~VulkanFuse() {
    mDescriptorSet = nullptr;
}

ErrorCode VulkanFuse::hotSwapConstBuffer(int bindingIndex, const void* data, size_t byteSize) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto cmdbuffer = std::shared_ptr<VulkanCommandPool::Buffer>(vkBn->getPool().allocBuffer());
    cmdbuffer->begin(0);

    // Find the matching const buffer offset
    for (auto& iter : mConstUniformOffset) {
        if (std::get<0>(iter) == bindingIndex) {
            auto bufSize = std::min(byteSize, std::get<1>(iter));
            // Create host-visible staging buffer
            auto hostBuf = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, bufSize, nullptr,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            auto ptr = hostBuf->map();
            ::memcpy(ptr, data, bufSize);
            hostBuf->unmap();
            // Copy to GPU const uniform buffer
            VkBufferCopy copy{};
            copy.size = bufSize;
            copy.dstOffset = std::get<2>(iter);
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdbuffer->get(), hostBuf->buffer(),
                           mConstUniformBuffer->buffer(), 1, &copy);
            auto fence = vkBn->getPool().submit(cmdbuffer->get());
            fence->wait();
            return NO_ERROR;
        }
    }
    for (auto& iter : mConstStorageOffset) {
        if (std::get<0>(iter) == bindingIndex) {
            auto bufSize = std::min(byteSize, std::get<1>(iter));
            auto hostBuf = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, bufSize, nullptr,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            auto ptr = hostBuf->map();
            ::memcpy(ptr, data, bufSize);
            hostBuf->unmap();
            VkBufferCopy copy{};
            copy.size = bufSize;
            copy.dstOffset = std::get<2>(iter);
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdbuffer->get(), hostBuf->buffer(),
                           mConstStorageBuffer->buffer(), 1, &copy);
            auto fence = vkBn->getPool().submit(cmdbuffer->get());
            fence->wait();
            return NO_ERROR;
        }
    }
    return NOT_SUPPORT;
}

ErrorCode VulkanFuse::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    for (int i=0; i<inputs.size(); ++i) {
        int binding = mInputBinding[i];
        auto tensorBuffer = vkBn->getBuffer(inputs[i]);
        mDescriptorSet->writeBuffer(tensorBuffer, binding);
    }
    for (int i=0; i<outputs.size(); ++i) {
        int binding = mOutputBinding[i];
        auto tensorBuffer = vkBn->getBuffer(outputs[i]);
        mDescriptorSet->writeBuffer(tensorBuffer, binding);
    }
    for (auto& iter : mConstStorageOffset) {
        mDescriptorSet->writeBuffer(mConstStorageBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
    }
    for (auto& iter : mConstUniformOffset) {
        mDescriptorSet->writeBuffer(mConstUniformBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
    }
    if (mNeedAutoTuning && !mOptimizedDispatch) {
        auto localSize = vkBn->autoTunePipeline(mPipeline.get(), mDescriptorSet, mGlobalSize);
        mPipeline->changePipeline(localSize);
        mGroupSize[0] = UP_DIV(mGlobalSize[0], localSize[0]);
        mGroupSize[1] = UP_DIV(mGlobalSize[1], localSize[1]);
        mGroupSize[2] = UP_DIV(mGlobalSize[2], localSize[2]);
        mNeedAutoTuning = false;
        // Re-create descriptor set after tuning (tuning may have invalidated it)
        mDescriptorSet = mPipeline->createSet();
        // Re-write descriptors
        for (int i=0; i<inputs.size(); ++i) {
            mDescriptorSet->writeBuffer(vkBn->getBuffer(inputs[i]), mInputBinding[i]);
        }
        for (int i=0; i<outputs.size(); ++i) {
            mDescriptorSet->writeBuffer(vkBn->getBuffer(outputs[i]), mOutputBinding[i]);
        }
        for (auto& iter : mConstStorageOffset) {
            mDescriptorSet->writeBuffer(mConstStorageBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
        }
        for (auto& iter : mConstUniformOffset) {
            mDescriptorSet->writeBuffer(mConstUniformBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
        }
    } else if (mOptimizedDispatch) {
        // Optimization: Use preferred workgroup size from shader
        // local_size (e.g. 16×16) for efficient GPU scheduling.
        // Group count = ceil(global_size / local_size).
        mGroupSize[0] = UP_DIV(mGlobalSize[0], mPreferredLocalSize[0]);
        mGroupSize[1] = UP_DIV(mGlobalSize[1], mPreferredLocalSize[1]);
        mGroupSize[2] = UP_DIV(mGlobalSize[2], mPreferredLocalSize[2]);
        mNeedAutoTuning = false;
    }
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    // Early-Z: Clamp dispatch to valid bounds, skipping entirely-outside workgroups.
    int dispatchX = mGroupSize[0], dispatchY = mGroupSize[1], dispatchZ = mGroupSize[2];
    if (mEarlyZ && mValidBounds.size() == 4 && mGlobalSize.size() >= 2) {
        int x0 = mValidBounds[0], y0 = mValidBounds[1];
        int x1 = mValidBounds[2], y1 = mValidBounds[3];
        int lsx = mGroupSize[0] > 0 ? UP_DIV(mGlobalSize[0], mGroupSize[0]) : 1;
        int lsy = mGroupSize[1] > 0 ? UP_DIV(mGlobalSize[1], mGroupSize[1]) : 1;
        // Compute dispatch origin and count for the valid region
        int gx0 = x0 / (mPreferredLocalSize[0] > 0 ? mPreferredLocalSize[0] : 16);
        int gy0 = y0 / (mPreferredLocalSize[1] > 0 ? mPreferredLocalSize[1] : 16);
        int gx1 = (x1 + mPreferredLocalSize[0] - 1) / (mPreferredLocalSize[0] > 0 ? mPreferredLocalSize[0] : 16);
        int gy1 = (y1 + mPreferredLocalSize[1] - 1) / (mPreferredLocalSize[1] > 0 ? mPreferredLocalSize[1] : 16);
        dispatchX = ALIMAX(1, gx1 - gx0);
        dispatchY = ALIMAX(1, gy1 - gy0);
    }
    vkCmdDispatch(cmdBuffer->get(), dispatchX, dispatchY, dispatchZ);
    return NO_ERROR;
}

} // namespace MNN

// Register VulkanFuseCreator for OpType_Extra
extern "C" __attribute__((visibility("default"))) void MNNVulkanFuseRegister() {
    MNN::VulkanBackend::addCreator(MNN::OpType_Extra, new MNN::VulkanFuseCreator);
}
