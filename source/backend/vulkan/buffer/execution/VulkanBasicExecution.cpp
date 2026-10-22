//
//  VulkanBasicExecution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBasicExecution.hpp"
#include "VulkanBackend.hpp"
#include "core/TensorUtils.hpp"
#include <atomic>
namespace MNN {
VulkanBasicExecutionDirect::VulkanBasicExecutionDirect(std::shared_ptr<VulkanBasicExecution> encoder) : Execution(encoder->backend()) {
    mEncoder = encoder;
    auto extra = static_cast<VulkanBackend *>(encoder->backend());
    mCmdBuffer.reset(const_cast<VulkanCommandPool::Buffer *>(extra->getPool().allocBuffer()));
}

ErrorCode VulkanBasicExecutionDirect::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto extra = static_cast<VulkanBackend *>(backend());
    auto code = mEncoder->onBeforeExecute(inputs, outputs);
    if (NO_ERROR != code) {
        return code;
    }
    // [TRACE] Per-frame execution marker: proves each Vulkan op's dispatch
    // actually runs. Output goes to the VulkanFuse trace file when armed.
    extern FILE* ispTraceSink();
    if (FILE* tf = ispTraceSink()) {
        fprintf(tf, "[EXEC] onExecute op=%s\n", mEncoder->getName().c_str());
        fflush(tf);
    }
    extra->pushCommand(mCmdBuffer->get());
    // [PROBE] When the ISP trace is armed: flush the queue so the recorded
    // dispatch + probe copy complete, then dump via afterExecute().
    // ONE-SHOT + selective: only the first 2 frames, and only for the ops
    // involved in the crop bisect (crop/normalize) — the probe path crashes
    // on some ops (SIGSEGV), so keep exposure minimal.
    {
        extern FILE* ispTraceSink();
        extern bool ispProbeFramesLeft(int);
        extern int ispProbeFrameTake();
        const std::string& nm = mEncoder->getName();
        // Only the front-of-chain ops — the wider per-op set SIGSEGVs inside
        // libMNN_Vulkan on this driver (probe staging across many op types).
        const bool wanted = (nm.find("isp.crop") != std::string::npos) ||
                            (nm.find("isp.normalize") != std::string::npos);
        if (ispTraceSink() != nullptr && wanted && ispProbeFramesLeft(2)) {
            ispProbeFrameTake();
            extra->finish();
            mEncoder->afterExecute(inputs, outputs);
        }
    }
    return NO_ERROR;
}


ErrorCode VulkanBasicExecutionDirect::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    {
        extern FILE* ispTraceSink();
        if (FILE* tf = ispTraceSink()) {
            fprintf(tf, "[EXEC] onResize op=%s\n", mEncoder->getName().c_str());
            fflush(tf);
        }
    }
    mCmdBuffer->begin(0);

    auto vkBn = static_cast<VulkanBackend*>(backend());
    for (auto input : inputs) {
        auto des = TensorUtils::getDescribeOrigin(input);
        if (0 == input->deviceId()) {
            continue;
        }
        auto vkTensor = (VulkanBuffer*)(input->deviceId());
        if (nullptr == vkTensor) {
            // The case occured if we don't need the content of input
            continue;
        }
        auto offset = des->offset;
        mCmdBuffer->barrierSource(vkTensor->buffer(), offset, vkBn->getTensorSize(input));
    }

#ifdef ENABLE_VULKAN_TIME_PROFILE
    ErrorCode code = NO_ERROR;
    {
        VulkanTimeProfileScope scope(vkBn->timeProfiler(), mCmdBuffer->get(), mEncoder->getName().c_str(),
                                     VulkanTimeProfiler::Kind::Execution);
        code = mEncoder->onEncode(inputs, outputs, mCmdBuffer.get());
    }
#else
    auto code = mEncoder->onEncode(inputs, outputs, mCmdBuffer.get());
#endif

    mCmdBuffer->end();
    return code;
}


VulkanBasicExecutionInDirect::VulkanBasicExecutionInDirect(std::shared_ptr<VulkanBasicExecution> encoder) : Execution(encoder->backend()) {
    mEncoder = encoder;
}
ErrorCode VulkanBasicExecutionInDirect::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    {
        extern FILE* ispTraceSink();
        if (FILE* tf = ispTraceSink()) {
            fprintf(tf, "[EXEC] InDirect::onExecute op=%s\n", mEncoder->getName().c_str());
            fflush(tf);
        }
    }
    auto code = mEncoder->onBeforeExecute(inputs, outputs);
    return code;
}
ErrorCode VulkanBasicExecutionInDirect::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto extra = static_cast<VulkanBackend *>(backend());
    auto cmdBuffer = extra->acquireIndirectSegmentForRecord();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    for (auto input : inputs) {
        auto vkTensor = (VulkanBuffer*)(input->deviceId());
        auto des = TensorUtils::getDescribeOrigin(input);
        if (nullptr == vkTensor) {
            // The case occured if we don't need the content of input
            continue;
        }
        auto offset = des->offset;
        cmdBuffer->barrierSource(vkTensor->buffer(), offset, vkBn->getTensorSize(input));
    }
    ErrorCode code = NO_ERROR;
#ifdef ENABLE_VULKAN_TIME_PROFILE
    {
        // The scope must finish before the indirect segment may be sealed.
        VulkanTimeProfileScope scope(vkBn->timeProfiler(), cmdBuffer->get(), mEncoder->getName().c_str(),
                                     VulkanTimeProfiler::Kind::Execution);
        code = mEncoder->onEncode(inputs, outputs, cmdBuffer.get());
    }
#else
    code = mEncoder->onEncode(inputs, outputs, cmdBuffer.get());
#endif
    if (NO_ERROR == code) {
        extra->finishIndirectRecordedOp();
    }
    return code;
}
} // namespace MNN
