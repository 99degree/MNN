// VulkanFuse.hpp — VulkanFuse execution class for custom SPIR-V compute shaders
#pragma once
#include "VulkanBasicExecution.hpp"
#include "component/VulkanPipeline.hpp"
#include "component/VulkanBuffer.hpp"
#ifdef MNN_BUILD_FOR_ANDROID
#include <android/log.h>
#endif
#include <MNN/MNNDefine.h>

#include "IspSpvLookup.hpp"

namespace MNN {

class VulkanFuse : public VulkanBasicExecution {
public:
    VulkanFuse(const Extra* extra, Backend* bn, int inputSize, int outputSize);
    // data (from embedded header lookup) when the Extra op has no spirv attr.
    virtual ~VulkanFuse();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    // Hot-swap: update const buffer data at runtime for live 3A adjustments.
    // bindingIndex: const buffer binding index (from Extra op attributes).
    // data: pointer to new float32 data.
    // byteSize: size of data in bytes.
    ErrorCode hotSwapConstBuffer(int bindingIndex, const void* data, size_t byteSize);
private:
    std::vector<int> mGroupSize;
    std::vector<int> mGlobalSize;
    std::vector<int> mInputBinding;
    std::vector<int> mOutputBinding;
    std::shared_ptr<VulkanBuffer> mConstStorageBuffer;
    std::shared_ptr<VulkanBuffer> mConstUniformBuffer;
    std::shared_ptr<VulkanBuffer> mConstStorageHostBuffer;
    std::shared_ptr<VulkanBuffer> mConstUniformHostBuffer;
    std::vector<std::tuple<int, size_t, size_t>> mConstStorageOffset;
    std::vector<std::tuple<int, size_t, size_t>> mConstUniformOffset;
    SharedPtr<VulkanPipeline> mPipeline;
    SharedPtr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::vector<int> mPreferredLocalSize;  // preferred workgroup size from attribute
    bool mNeedAutoTuning = false;
    bool mOptimizedDispatch = false;
    bool mEarlyZ = false;  // skip workgroups outside valid image bounds
    std::vector<int> mValidBounds; // {x0, y0, x1, y1} pixel coords (or empty = no cull)
    bool mFp16Consts = false; // pack const buffers as FP16 (halves bandwidth)
    std::vector<uint8_t> mFp16DataStorage; // keeps FP16 packed data alive during merge
    std::string mType;   // Extra op type string (e.g. "isp.ispc_stats")
    bool mReduce = false; // true for single-workgroup reduction extras (reduce_keepdims attr)
    bool mReducePatched = false; // const W/H/C patched once from actual input dims
    bool mElementwise = false;   // true for elementwise isp.* ops (output shape = input shape)
    bool mElementwisePatched = false; // const W/H patched once for elementwise ops
    // Runtime hot-swap: host-visible const buffer + GPU-side buffer for live 3A updates.
    std::shared_ptr<VulkanBuffer> mRuntimeHostBuffer;   // host-visible staging
    std::shared_ptr<VulkanBuffer> mRuntimeGpuBuffer;    // GPU-side const uniform/storage
    size_t mRuntimeBufferSize = 0;
};

class VulkanFuseCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto extra = op->main_as_Extra();
        if (nullptr == extra) return nullptr;
        if (nullptr == extra->attr()) return nullptr;
        if(extra->type()->str() == "ExtraConvolution2DPrelu") return nullptr;
        // Reject Extra ops without a "spirv" attribute — the VulkanFuse
        // constructor requires valid SPIR-V bytecode to build a compute
        // pipeline; constructing without it would leave the object in an
        // invalid state (mPipeline==nullptr) and crash in onEncode().
        // Returning nullptr here lets Pipeline::createExecutionWithExternal
        // fall through to the CPU backup backend, which can then either
        // run a CPU implementation or return NOT_SUPPORT.
        bool hasSpirv = false;
        for (int i = 0; i < extra->attr()->size(); ++i) {
            if (extra->attr()->GetAs<Attribute>(i)->key()->str() == "spirv") {
                hasSpirv = true;
                break;
            }
        }
        if (!hasSpirv) {
            // [ISP EMBEDDED SPIRV] Fallback: check embedded shader table
            if (extra->type() && extra->type()->str().rfind("isp.", 0) == 0) {
                auto spvData = lookupIspSpv(extra->type()->str());
                if (spvData.data != nullptr && spvData.size > 0) {
                    fprintf(stderr, "[VulkanFuse] ACCEPT '%s' (embedded SPIR-V, %zu bytes) — %d inputs, %d outputs\n",
                              extra->type()->str().c_str(), spvData.size, (int)inputs.size(), (int)outputs.size());
                    fflush(stderr);
                    return new VulkanFuse(extra, backend, (int)inputs.size(), (int)outputs.size()); // embedded SPIR-V lookup happens in constructor
                }
            }
            fprintf(stderr, "[VulkanFuse] REJECT '%s' (no SPIR-V) — %d inputs, %d outputs\n",
                      extra->type()->str().c_str(), (int)inputs.size(), (int)outputs.size());
            fflush(stderr);
            return nullptr;
        }
        fprintf(stderr, "[VulkanFuse] ACCEPT '%s' — %d inputs, %d outputs\n",
                  extra->type()->str().c_str(), (int)inputs.size(), (int)outputs.size());
        fflush(stderr);
        // Note: VulkanFuse executes user-provided SPIR-V with no metadata about
        // expected tensor layout. If an Extra op lands on tensors whose format
        // (NC4HW4 vs NCHW) the SPIR-V doesn't handle, results will be wrong
        // without any error. There is no Creator-side way to detect this from
        // the flatbuffer schema; callers should ship SPIR-V that matches the
        // graph's tensor format or use a backend that converts layouts.
        return new VulkanFuse(extra, backend, (int)inputs.size(), (int)outputs.size());
    }
};

} // namespace MNN
