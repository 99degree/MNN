//
//  VulkanFuse.cpp
//  MNN
//
//  Created by MNN on 2023/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
// Explicitly include buffer backend's VulkanBackend.hpp (has getPipelineFactory/getBuffer)
// to avoid image/backend shadowing it.
#include "../backend/VulkanBackend.hpp"
#include "VulkanBasicExecution.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanFuse.hpp"
#include "IspSpvLookup.hpp"
#include "core/OpCommonUtils.hpp"
#include <stdio.h>
#include <chrono>
#include <cstdio>
#include <stdlib.h>
namespace MNN {

// Perf-build gate for the per-dispatch VulkanFuse diagnostics. In a full-speed
// perf run these fprintf(stderr) calls (~14 ops x ~10 lines per frame on a 30fps
// camera stream) dominate host overhead; they are opt-in via ISP_DEBUG_VLOG so
// the default build is silent + fast. Default OFF.
static inline bool ispVlog() {
    static const bool v = (getenv("ISP_DEBUG_VLOG") &&
                           strcmp(getenv("ISP_DEBUG_VLOG"), "0") != 0);
    return v;
}
#define ISP_VLOG(...) do { if (ispVlog()) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while (0)

VulkanFuse::VulkanFuse(const Extra* extra, Backend* bn, int inputSize, int outputSize) : VulkanBasicExecution(bn) {
    auto vkBn = static_cast<VulkanBackend*>(bn);
    auto factory = vkBn->getPipelineFactory();
    if (extra->type()) mType = extra->type()->str();
#ifdef ENABLE_VULKAN_TIME_PROFILE
    // Name the per-op GPU timing record after the real isp.* type ("isp.normalize"
    // etc.) so each stage is visible individually in the profile string instead
    // of being collapsed into a single "Extra" bucket. VulkanBackend::onCreate
    // preserves this name for OpType_Extra ops (does not overwrite it).
    if (extra && extra->type()) {
        setName(extra->type()->c_str());
    }
#endif
    mOutputBinding.resize(outputSize);
    mInputBinding.resize(inputSize);
    mGroupSize.resize(3);
    mGlobalSize.resize(3);
    // Detect single-workgroup reduction extras (reduce_keepdims attr)
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "reduce_keepdims" && (attr->b() || attr->i() != 0)) {
            mReduce = true;
            break;
        }
    }
    // Detect elementwise extras (output shape copies input shape)
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "elementwise" && (attr->b() || attr->i() != 0)) {
            mElementwise = true;
            break;
        }
    }
    // Find shader
    const uint8_t* data = nullptr;
    size_t dataSize = 0;
    // [ISP EMBEDDED SPIRV] Check embedded lookup first
    if (extra->type() && extra->type()->str().rfind("isp.", 0) == 0) {
        auto spvData = lookupIspSpv(extra->type()->str());
        if (spvData.data != nullptr && spvData.size > 0) {
            data = spvData.data;
            dataSize = spvData.size;
        }
    }
    // If not found in embedded lookup, check Extra op attributes
    if (data == nullptr) {
        for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "spirv") {
            // Try tensor format first (int8s)
            if (attr->tensor() && attr->tensor()->int8s() && attr->tensor()->int8s()->size() > 0) {
                data = (uint8_t*)attr->tensor()->int8s()->data();
                dataSize = attr->tensor()->int8s()->size();
                break;
            }
            // Fallback: check string list (ONNX STRING attribute converted to string list)
            if (attr->list() && attr->list()->s() && attr->list()->s()->size() > 0) {
                const ::flatbuffers::String* fbStr = attr->list()->s()->GetAsString(0);
                if (fbStr) {
                    data = (const uint8_t*)fbStr->c_str();
                    dataSize = fbStr->size();
                }
                break;
            }
            // Fallback: check single string attribute
            if (attr->s() && attr->s()->size() > 0) {
                auto str = attr->s()->str();
                data = (const uint8_t*)str.c_str();
                dataSize = str.size();
                break;
            }
        }
    }
    }
    // Caller (VulkanFuseCreator::onCreate) is responsible for validating
    // that SPIR-V is present before constructing VulkanFuse. Reaching here
    // with data==nullptr is a programmer error; fall through to the rest of
    // the constructor which will fail loudly at createComputePipeline().

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
    mConstUniformHostBuffer = std::get<2>(uniforms);
    auto storages = merge(constStoragePtrs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    mConstStorageOffset = std::get<0>(storages);
    mConstStorageBuffer = std::get<1>(storages);
    mConstStorageHostBuffer = std::get<2>(storages);
    cmdbuffer->end();
    auto fence = vkBn->getPool().submit(cmdbuffer->get());

    // [ISP] Direct-emit fallback. The converter's IspOpConverter (run at
    // optimizeLevel=0) copies the ONNX isp.* op's attributes verbatim, but the
    // direct-emit path (OnnxGraphComposer) emits the isp.* ops WITHOUT the
    // input/const/global_size binding attributes that VulkanFuse normally reads.
    // lookupIspSpv() still resolves the (SPIR-V-baked) shader by type, but
    // `types`/`mInputBinding`/`mOutputBinding` come back EMPTY -> the
    // descriptor-set layout built from `types` has ZERO bindings, and the first
    // DescriptorSet::writeBuffer(bind) reads mBufferTypes[bind] out-of-bounds,
    // yielding a garbage descriptor type (0xFFFFFFFF). Mesa/freedreno then
    // dereferences a null descriptor-set-layout entry inside
    // vkUpdateDescriptorSets -> SIGSEGV, fault_addr=0x18.
    //
    // All 50 embedded isp.* shaders share an identical, verified convention
    // (dumped from every isp_*_spv.h via spirv-dis): LocalSize 16x16x1;
    // binding 0 = const/param SSBO (read-only), binding 1 = input frame
    // (NonWritable StorageBuffer), binding 2 = output frame (NonReadable
    // StorageBuffer); all BufferBlock/STORAGE_BUFFER. Populate the layout and
    // the tensor->binding map from this convention so the ctor-built
    // descriptor set is valid. Extra scalar inputs (e.g. isp.normalize's
    // sensor_max) have no SPIR-V storage binding -> bound to a reserved
    // binding 3 the shader never references, so the descriptor is valid and the
    // shader never dereferences them.
    if (types.empty() && !mType.empty() && mType.rfind("isp.", 0) == 0) {
        int maxBind = 3;
        types.assign(maxBind + 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // 0,1,2,3
        for (int i = 0; i < inputSize; ++i) {
            mInputBinding[i] = (i == 0) ? 1 : 3; // frame->bind1; extras->dummy bind3
        }
        for (int i = 0; i < outputSize; ++i) {
            mOutputBinding[i] = (i == 0) ? 2 : 3; // frame->bind2
        }
        // binding 0 param SSBO: allocate a zero-filled, host-visible storage
        // buffer so the shader reads valid (zero) params -> bounds-checks all
        // work-items out -> no-op dispatch (no null-buffer GPU fault in Mesa).
        const size_t cbsz = 256;
        mConstStorageBuffer = std::make_shared<VulkanBuffer>(
            vkBn->getMemoryPool(), false, cbsz, nullptr,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        mConstStorageHostBuffer = mConstStorageBuffer;
        auto* cbp = (uint8_t*)mConstStorageBuffer->map();
        if (cbp) { ::memset(cbp, 0, cbsz); mConstStorageBuffer->unmap(); }
        mConstStorageOffset.clear();
        mConstStorageOffset.emplace_back(std::make_tuple(0, cbsz, size_t(0)));
        mPreferredLocalSize = {16, 16, 1};
        mOptimizedDispatch = true;
        mNeedAutoTuning = false;   // fixed LocalSize; no pipeline rebuild
    }
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

    // When fp16_consts=true the const buffer holds PACKED FP16 values
    // (shader reads via unpackHalf2x16). Convert the input floats to FP16
    // so the hot-swapped bytes match the buffer layout.
    std::vector<uint8_t> fp16Packed;
    const void* srcData = data;
    size_t srcSize = byteSize;
    if (mFp16Consts && (byteSize % sizeof(float)) == 0) {
        int count = (int)(byteSize / sizeof(float));
        std::vector<uint16_t> fp16(count);
        const float* fdata = (const float*)data;
        for (int j = 0; j < count; j++) {
            float v = fdata[j];
            uint32_t f = *((uint32_t*)&v);
            uint32_t sign = (f >> 16) & 0x8000;
            int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
            uint32_t mantissa = (f >> 13) & 0x3FF;
            if (exponent <= 0) { fp16[j] = (uint16_t)sign; }
            else if (exponent >= 31) { fp16[j] = (uint16_t)(sign | 0x7BFF); }
            else { fp16[j] = (uint16_t)(sign | (exponent << 10) | mantissa); }
        }
        fp16Packed.resize(count * sizeof(uint16_t));
        ::memcpy(fp16Packed.data(), fp16.data(), count * sizeof(uint16_t));
        srcData = fp16Packed.data();
        srcSize = count * sizeof(uint16_t);
    }

    // Find the matching const buffer offset
    for (auto& iter : mConstUniformOffset) {
        if (std::get<0>(iter) == bindingIndex) {
            auto bufSize = std::min(srcSize, std::get<1>(iter));
            // Create host-visible staging buffer
            auto hostBuf = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, bufSize, nullptr,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            auto ptr = hostBuf->map();
            ::memcpy(ptr, srcData, bufSize);
            hostBuf->unmap();
            // Copy to GPU const uniform buffer
            VkBufferCopy copy{};
            copy.size = bufSize;
            copy.dstOffset = std::get<2>(iter);
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdbuffer->get(), hostBuf->buffer(),
                           mConstUniformBuffer->buffer(), 1, &copy);
            cmdbuffer->end();
            auto fence = vkBn->getPool().submit(cmdbuffer->get());
            fence->wait();
            return NO_ERROR;
        }
    }
    for (auto& iter : mConstStorageOffset) {
        if (std::get<0>(iter) == bindingIndex) {
            auto bufSize = std::min(srcSize, std::get<1>(iter));
            auto hostBuf = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, bufSize, nullptr,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            auto ptr = hostBuf->map();
            ::memcpy(ptr, srcData, bufSize);
            hostBuf->unmap();
            VkBufferCopy copy{};
            copy.size = bufSize;
            copy.dstOffset = std::get<2>(iter);
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdbuffer->get(), hostBuf->buffer(),
                           mConstStorageBuffer->buffer(), 1, &copy);
            cmdbuffer->end();
            auto fence = vkBn->getPool().submit(cmdbuffer->get());
            fence->wait();
            return NO_ERROR;
        }
    }
    cmdbuffer->end();
    return NOT_SUPPORT;
}

ErrorCode VulkanFuse::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    ISP_VLOG("[VulkanFuse] onEncode type=%s inputs=%zu outputs=%zu global=[%d,%d,%d] group=[%d,%d,%d] constOff=%zu\n",
        mType.c_str(), inputs.size(), outputs.size(),
        mGlobalSize[0], mGlobalSize[1], mGlobalSize[2],
        mGroupSize[0], mGroupSize[1], mGroupSize[2],
        mConstStorageOffset.size() + mConstUniformOffset.size());
    fflush(stderr);
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto* t = inputs[i];
        int b = (i < mInputBinding.size()) ? mInputBinding[i] : -1;
        ISP_VLOG("[VulkanFuse]   in[%zu] bind=%d dims=%d size=%zu\n", i, b,
            t ? (int)t->buffer().dimensions : -1, t ? (size_t)t->elementSize() : 0);
    }
    fflush(stderr);
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto* t = outputs[i];
        int b = (i < mOutputBinding.size()) ? mOutputBinding[i] : -1;
        ISP_VLOG("[VulkanFuse]   out[%zu] bind=%d dims=%d size=%zu\n", i, b,
            t ? (int)t->buffer().dimensions : -1, t ? (size_t)t->elementSize() : 0);
    }
    fflush(stderr);
    for (auto& iter : mConstStorageOffset) {
        ISP_VLOG("[VulkanFuse]   constStorage bind=%d off=%zu len=%zu\n",
            std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
    }
    for (auto& iter : mConstUniformOffset) {
        ISP_VLOG("[VulkanFuse]   constUniform bind=%d off=%zu len=%zu\n",
            std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
    }
    fflush(stderr);
    // Dump const buffer first 8 floats (uniforms: W,H,gain,center_th,...)
    // NOTE: mConstStorageBuffer is DEVICE-local (unreadable on CPU); read the
    // HOST staging buffer which holds the same data.
    auto dumpConstHost = [&](const std::shared_ptr<VulkanBuffer>& hostBuf,
                              const char* tag) {
        if (!hostBuf) return;
        auto* p = hostBuf->map();
        if (p) {
            const float* f = (const float*)p;
            ISP_VLOG("[VulkanFuse]   %s[0..7]=%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                tag, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
            fflush(stderr);
            hostBuf->unmap();
        }
    };
    dumpConstHost(mConstStorageHostBuffer, "constStorageHost");
    dumpConstHost(mConstUniformHostBuffer, "constUniformHost");
    // ── Dynamic-size derivation ──
    // If the baked global_size is invalid (0/negative — from dynamic ONNX input
    // at convert time), derive the dispatch size from the ACTUAL input tensor
    // dims at runtime, and patch the const buffer's first two floats (W, H).
    // dims==1 (scalar op): treat as 1×1. dims==2: [H,W]. dims>=4: [N,C,H,W].
    int tw = 0, th = 0, tc = 0;
    if (!inputs.empty() && inputs[0] && inputs[0]->buffer().dimensions >= 1) {
        int dims = inputs[0]->buffer().dimensions;
        if (dims == 1) {
            tw = 1; th = 1;
        } else if (dims == 2) {
            th = inputs[0]->buffer().dim[0].extent;
            tw = inputs[0]->buffer().dim[1].extent;
        } else {
            tw = inputs[0]->buffer().dim[dims-1].extent;   // W
            th = inputs[0]->buffer().dim[dims-2].extent;   // H
            if (dims >= 4) tc = inputs[0]->buffer().dim[1].extent;  // C
        }
    }
    if (mGlobalSize[0] <= 0 || mGlobalSize[1] <= 0) {
        if (tw > 0 && th > 0) {
            mGlobalSize[0] = tw;
            mGlobalSize[1] = th;
            mGlobalSize[2] = 1;
            mNeedAutoTuning = true;
        } else {
            // Never dispatch 0 workgroups — vkCmdDispatch requires ≥1. Fall
            // back to a single thread so the shader at least runs.
            mGlobalSize[0] = 1;
            mGlobalSize[1] = 1;
            mGlobalSize[2] = 1;
        }
    }
    // Elementwise extras: the converter bakes the stride-2 global_size
    // (demosaic convention) which is wrong for standalone elementwise blocks
    // (BLC/fcs/ee/gamma/lsc/display/...). The REAL dispatch size and the
    // shader's const W,H must match the ACTUAL input tensor dims — otherwise
    // the shader early-returns on `x >= w || y >= h` and writes nothing.
    // Patch BOTH: override global_size from input dims AND hot-swap the
    // const buffer's first two floats (W,H).
    if (mElementwise && !mElementwisePatched && tw > 0 && th > 0) {
        ISP_VLOG("[VulkanFuse] ELEMENTWISE patch: tw=%d th=%d glob=[%d,%d,%d]\n",
            tw, th, mGlobalSize[0], mGlobalSize[1], mGlobalSize[2]);
        fflush(stderr);
        if (mGlobalSize[0] != tw || mGlobalSize[1] != th) {
            mGlobalSize[0] = tw;
            mGlobalSize[1] = th;
            mGlobalSize[2] = 1;
            mNeedAutoTuning = true;
        }
        float v[2] = { (float)tw, (float)th };
        ISP_VLOG("[VulkanFuse] ELEMENTWISE hotSwap begin\n");
        fflush(stderr);
        ErrorCode hs = hotSwapConstBuffer(0, v, sizeof(v));
        ISP_VLOG("[VulkanFuse] ELEMENTWISE hotSwap done rc=%d\n", (int)hs);
        fflush(stderr);
        mElementwisePatched = true;
    }
    // Reduction extras: baked const {W,H(,C)} is 0 from dynamic input. Patch
    // ONCE (first encode) from the actual input tensor so the stride/reduce
    // loops run over the real image. ispc_stats const = {W,H,C};
    // calib_stats/af_focus = {W,H}. (Dims are stable for a fixed sensor.)
    if (mReduce && !mReducePatched && tw > 0 && th > 0) {
        if (mType == "isp.ispc_stats" && tc > 0) {
            float v[3] = { (float)tw, (float)th, (float)tc };
            hotSwapConstBuffer(0, v, sizeof(v));
        } else {
            float v[2] = { (float)tw, (float)th };
            hotSwapConstBuffer(0, v, sizeof(v));
        }
        mReducePatched = true;
    }
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
        // VulkanFuse dispatches a FUSED compute shader whose workgroup size is
        // FIXED (OpExecutionMode LocalSize) in the SPIR-V — 16x16x1 for all
        // isp.* ops, which matches the mPreferredLocalSize default populated
        // from the group_size attr (or the {16,16,1} fallback). There is
        // nothing to auto-tune: the spec-constant local size MUST equal the
        // shader's baked %gl_WorkGroupSize. The legacy autoTunePipeline()
        // path called changePipeline() to rebuild the pipeline with candidate
        // local sizes — but it re-derived the pipeline's descriptor-set layout
        // and corrupted mBufferTypes (size 1, garbage type 0xFFFFFFFF), which
        // made the subsequent createSet()+rebind deref a null layout entry in
        // Mesa/freedreno (fault_addr=0x18, SIGSEGV). Reuse the valid
        // constructor-built descriptor set (mDescriptorSet, whose layout spans
        // every input/output binding) and compute the dispatch group count from
        // the shader-fixed local size.
        int lx = mPreferredLocalSize[0] > 0 ? mPreferredLocalSize[0] : 16;
        int ly = mPreferredLocalSize[1] > 0 ? mPreferredLocalSize[1] : 16;
        int lz = mPreferredLocalSize[2] > 0 ? mPreferredLocalSize[2] : 1;
        mGroupSize[0] = UP_DIV(mGlobalSize[0], lx);
        mGroupSize[1] = UP_DIV(mGlobalSize[1], ly);
        mGroupSize[2] = UP_DIV(mGlobalSize[2], lz);
        mNeedAutoTuning = false;
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
    auto t_disp0 = std::chrono::high_resolution_clock::now();
    vkCmdDispatch(cmdBuffer->get(), dispatchX, dispatchY, dispatchZ);
    auto t_disp1 = std::chrono::high_resolution_clock::now();
    auto disp_us = std::chrono::duration_cast<std::chrono::microseconds>(t_disp1 - t_disp0).count();
    ISP_VLOG("[VulkanFuse] DISPATCH type=%s glob=[%d,%d,%d] group=[%d,%d,%d] earlyZ=%d bounds=[%d,%d,%d,%d] submit_us=%lld\n",
        mType.c_str(), mGlobalSize[0], mGlobalSize[1], mGlobalSize[2],
        mGroupSize[0], mGroupSize[1], mGroupSize[2],
        mEarlyZ ? 1 : 0,
        mValidBounds.size() == 4 ? mValidBounds[0] : -1,
        mValidBounds.size() == 4 ? mValidBounds[1] : -1,
        mValidBounds.size() == 4 ? mValidBounds[2] : -1,
        mValidBounds.size() == 4 ? mValidBounds[3] : -1,
        (long long)disp_us);
    // Per-op GPU workload reporting (writes to stderr → logcat)
    static thread_local int _log_counter = 0;
    if (::getenv("ISP_DEBUG_VLOG") || (_log_counter < 40)) {
        fprintf(stderr, "[VulkanFuse-PERF] type=%s W=%d H=%d groups=[%dx%dx%d] wg=%dx%dx%d submit_us=%lld pixels=%d\n",
            mType.c_str(), mGlobalSize[0], mGlobalSize[1],
            mGroupSize[0], mGroupSize[1], mGroupSize[2],
            dispatchX, dispatchY, dispatchZ,
            (int)disp_us, mGlobalSize[0] * mGlobalSize[1]);
        fflush(stderr);
        // Also emit via stderr (goes to logcat on Android)
        _log_counter++;
    }
    return NO_ERROR;
}

} // namespace MNN

// Register VulkanFuseCreator for OpType_Extra
extern "C" __attribute__((visibility("default"))) void MNNVulkanFuseRegister() {
    MNN::VulkanBackend::addCreator(MNN::OpType_Extra, new MNN::VulkanFuseCreator);
}
