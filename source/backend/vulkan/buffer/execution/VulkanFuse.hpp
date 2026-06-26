// VulkanFuse.hpp — VulkanFuse execution class for custom SPIR-V compute shaders
#pragma once
#include "VulkanBasicExecution.hpp"
#include "component/VulkanPipeline.hpp"
#include "component/VulkanBuffer.hpp"
#include <MNN/MNNDefine.h>

namespace MNN {

class VulkanFuse : public VulkanBasicExecution {
public:
    VulkanFuse(const Extra* extra, Backend* bn, int inputSize, int outputSize);
    virtual ~VulkanFuse();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
private:
    std::vector<int> mGroupSize;
    std::vector<int> mGlobalSize;
    std::vector<int> mInputBinding;
    std::vector<int> mOutputBinding;
    std::shared_ptr<VulkanBuffer> mConstStorageBuffer;
    std::shared_ptr<VulkanBuffer> mConstUniformBuffer;
    std::vector<std::tuple<int, size_t, size_t>> mConstStorageOffset;
    std::vector<std::tuple<int, size_t, size_t>> mConstUniformOffset;
    SharedPtr<VulkanPipeline> mPipeline;
    SharedPtr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::vector<int> mPreferredLocalSize;  // preferred workgroup size from attribute
    bool mNeedAutoTuning = false;
    bool mOptimizedDispatch = false;
};

class VulkanFuseCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto extra = op->main_as_Extra();
        if (nullptr == extra) return nullptr;
        if (nullptr == extra->attr()) return nullptr;
        if(extra->type()->str() == "ExtraConvolution2DPrelu") return nullptr;
        return new VulkanFuse(extra, backend, (int)inputs.size(), (int)outputs.size());
    }
};

} // namespace MNN
