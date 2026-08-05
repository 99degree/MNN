//
//  VulkanReduce.cpp
//  MNN
//
//  Created by MNN on 2020/03/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanReduce.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Macro.h"
namespace MNN {
struct constBuffer {
    int w;//inside
    int h;//axis
    int c;//outside
    float k;//For mean
    int reduceAxis;
};
#define MAX_VALUE 10001.f
VulkanReduce::VulkanReduce(const std::string& name, const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBn = (VulkanBackend*)backend();
    mOp = op;
    mPipeline = vkBn->getPipeline(name, {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    });
    mConstBuffer = vkBn->allocUniform();
    mDescriptorSet.reset(mPipeline->createSet());
}
VulkanReduce::~VulkanReduce() {
    auto vkBn = (VulkanBackend*)backend();
    vkBn->recycleUniform(mConstBuffer);
}

ErrorCode VulkanReduce::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto inputTensor = vkBn->getBuffer(inputs[0]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    auto ptr = reinterpret_cast<constBuffer*>(mConstBuffer->map());
    ::memset(ptr, 0, sizeof(constBuffer));
    auto reduce = mOp->main_as_ReductionParam();
    auto dimParam = reduce->dim();
    int rank = inputs[0]->dimensions();
    // Resolve axes: handle null dim (reduce all), negative axes, and multi-axis.
    // Caller (Creator) has validated that axes are sorted+contiguous; here we
    // just normalize negatives and compute the combined reduce window.
    int axisCount = (nullptr == dimParam) ? rank : dimParam->size();
    int axes[MNN_MAX_TENSOR_DIM];
    if (nullptr == dimParam) {
        for (int i = 0; i < rank; ++i) {
            axes[i] = i;
        }
    } else {
        for (int i = 0; i < axisCount; ++i) {
            int a = dimParam->data()[i];
            if (a < 0) {
                a += rank;
            }
            axes[i] = a;
        }
    }
    int minAxis = rank;
    int maxAxis = -1;
    for (int i = 0; i < axisCount; ++i) {
        if (axes[i] < minAxis) {
            minAxis = axes[i];
        }
        if (axes[i] > maxAxis) {
            maxAxis = axes[i];
        }
    }
    // Combined axis length: product of all reduced dims (sorted, contiguous by Creator check).
    int axis = 1;
    for (int i = minAxis; i <= maxAxis; ++i) {
        axis *= inputs[0]->length(i);
    }
    int inside = 1;
    for (int i = maxAxis + 1; i < rank; ++i) {
        inside *= inputs[0]->length(i);
    }
    int outside = 1;
    for (int i = 0; i < minAxis; ++i) {
        outside *= inputs[0]->length(i);
    }
    ptr->c = outside;
    ptr->h = axis;
    ptr->w = inside;
    ptr->k = 1.0f/(float)axis;
    auto total = outside * inside;
    int outsideParallel = 1;
    if (total >= 256) {
        ptr->reduceAxis = 1;
        outsideParallel = 256;
    } else if (total < 16) {
        ptr->reduceAxis = 256;
        outsideParallel = 1;
    } else {
        ptr->reduceAxis = 16;
        outsideParallel = 16;
    }
    //MNN_PRINT("o, i, axis: %d - %d - %d => op %d, ra %d\n", outside, inside, axis, outsideParallel, ptr->reduceAxis);
    mConstBuffer->unmap();
    // Encode
    mDescriptorSet->writeBuffer(outputTensor, 0);
    mDescriptorSet->writeBuffer(inputTensor, 1);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
    cmdBuffer->barrierSource(inputTensor);
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, outsideParallel), 1, 1);
    return NO_ERROR;
}

static std::string _getShaderName(const Op* op, bool isInt, bool useFP16) {
    std::string result = "glsl_reduce_";
    if (isInt) {
        result += "int_";
    }
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_SUM:
            result += "SUM";
            break;
        case ReductionType_MEAN:
            result += "MEAN";
            break;
        case ReductionType_MAXIMUM:
            result += "VMAX";
            break;
        case ReductionType_MINIMUM:
            result += "VMIN";
            break;
        case ReductionType_PROD:
            result += "PROD";
            break;
        default:
            return "";
    }
    result += "_";
    if (useFP16 && !isInt) {
        result += "FP16_";
    }
    result += "comp";
    return result;
}
class VulkanReduceCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        auto input0 = inputs[0];
        bool isint = input0->getType().code == halide_type_int;
        auto vkBn = static_cast<VulkanBackend*>(backend);
        auto shader = _getShaderName(op, isint, vkBn->useFP16());
        if (shader.empty()) {
            return nullptr;
        }
        // Validate reduction axes for Vulkan kernel constraints.
        // The reduce shader performs a single contiguous-window reduction per
        // (outside, inside) pair, so it requires axes to be sorted and
        // contiguous. Non-contiguous multi-axis reduce must fall back to CPU
        // (CPUReduction handles arbitrary axes correctly).
        auto reduce = op->main_as_ReductionParam();
        auto dimParam = reduce->dim();
        int rank = input0->dimensions();
        int axisCount = (nullptr == dimParam) ? rank : dimParam->size();
        if (axisCount <= 0 || axisCount > rank) {
            return nullptr;
        }
        int axes[MNN_MAX_TENSOR_DIM];
        if (nullptr == dimParam) {
            for (int i = 0; i < rank; ++i) {
                axes[i] = i;
            }
        } else {
            for (int i = 0; i < axisCount; ++i) {
                int a = dimParam->data()[i];
                if (a < 0) {
                    a += rank;
                }
                if (a < 0 || a >= rank) {
                    return nullptr;
                }
                axes[i] = a;
            }
        }
        // Insertion sort (axisCount is small).
        for (int i = 1; i < axisCount; ++i) {
            int key = axes[i];
            int j = i - 1;
            while (j >= 0 && axes[j] > key) {
                axes[j + 1] = axes[j];
                --j;
            }
            axes[j + 1] = key;
        }
        // Require sorted+contiguous.
        for (int i = 1; i < axisCount; ++i) {
            if (axes[i] != axes[i - 1] + 1) {
                return nullptr;
            }
        }
        return new VulkanReduce(shader, op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Reduction, new VulkanReduceCreator);
    return true;
}();
}