//
//  VulkanInterp.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanInterp.hpp"

namespace MNN {

VulkanInterp::VulkanInterp(const Op* op, Backend* bn) : VulkanResize(bn, 1, 1, op->main_as_Interp()->resizeType()) {
    auto interpParam = op->main_as_Interp();
    mCordTransform[0] = interpParam->widthScale();
    mCordTransform[1] = interpParam->widthOffset();
    mCordTransform[2] = interpParam->heightScale();
    mCordTransform[3] = interpParam->heightOffset();
}

VulkanInterp::~VulkanInterp() {
}

ErrorCode VulkanInterp::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    encodeImpl(input, output, mCordTransform, cmdBuffer);

    return NO_ERROR;
}

class VulkanInterpCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        // Graceful fallback: Vulkan Interp only supports FLOAT. Return nullptr so
        // the Session routes the op to the CPU backend instead of aborting via assert.
        if (outputs[0]->getType().code != halide_type_float) {
            return nullptr;
        }
        return new VulkanInterp(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Interp, new VulkanInterpCreator);
    return true;
}();

} // namespace MNN
