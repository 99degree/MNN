// ShapeExtra.cpp — SizeComputer for OpType_Extra
// Handles SpaceToDepthEx custom Extra op for ISP pipeline
#include "shape/SizeComputer.hpp"
#include "core/OpCommonUtils.hpp"
using namespace MNN;

class ShapeExtra : public SizeComputer {
public:
    virtual bool onComputeSize(const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto extra = op->main_as_Extra();
        if (nullptr == extra || nullptr == extra->attr()) {
            return false;
        }
        // Find global_size or blocksize attribute
        for (int i = 0; i < extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "global_size") {
                auto tensor = attr->tensor();
                if (tensor && tensor->int32s()) {
                    int n = tensor->int32s()->data()[0];
                    int h = tensor->int32s()->size() > 1 ? tensor->int32s()->data()[1] : n;
                    int w = tensor->int32s()->size() > 2 ? tensor->int32s()->data()[2] : h;
                    outputs[0]->buffer().dim[0].extent = 1;
                    outputs[0]->buffer().dim[1].extent = h;
                    outputs[0]->buffer().dim[2].extent = w;
                    outputs[0]->buffer().dimensions = 3;
                    TensorUtils::getDescribe(outputs[0])->dimensionFormat = 
                        TensorUtils::getDescribe(inputs[0])->dimensionFormat;
                    return true;
                }
            }
        }
        // Fallback: match input dims
        if (inputs[0]->buffer().dimensions >= 2) {
            outputs[0]->buffer().dimensions = inputs[0]->buffer().dimensions;
            for (int d = 0; d < inputs[0]->buffer().dimensions; d++) {
                outputs[0]->buffer().dim[d].extent = inputs[0]->buffer().dim[d].extent;
            }
            return true;
        }
        return false;
    }
};
REGISTER_SHAPE(ShapeExtra, OpType_Extra);
