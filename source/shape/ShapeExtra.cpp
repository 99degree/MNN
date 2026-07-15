// ShapeExtra.cpp — SizeComputer for OpType_Extra
#include "shape/SizeComputer.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {

class ShapeExtra : public SizeComputer {
public:
    virtual bool onComputeSize(const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto extra = op->main_as_Extra();
        if (nullptr == extra || nullptr == extra->attr()) {
            return false;
        }
        // First check for explicit output_shape attribute: [N, C, H, W]
        // Can be stored as INTS list (from ONNX conversion) or as tensor BlobT
        for (int i = 0; i < extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "output_shape") {
                // Check INTS list first (most common from ONNX converter)
                if (attr->list() && attr->list()->i() && attr->list()->i()->size() == 4) {
                    auto data = attr->list()->i()->data();
                    outputs[0]->buffer().dimensions = 4;
                    outputs[0]->buffer().dim[0].extent = data[0];
                    outputs[0]->buffer().dim[1].extent = data[1];
                    outputs[0]->buffer().dim[2].extent = data[2];
                    outputs[0]->buffer().dim[3].extent = data[3];
                    return true;
                }
                // Fall back to tensor BlobT
                if (attr->tensor() && attr->tensor()->int32s() && attr->tensor()->int32s()->size() == 4) {
                    auto data = attr->tensor()->int32s()->data();
                    outputs[0]->buffer().dimensions = 4;
                    outputs[0]->buffer().dim[0].extent = data[0];
                    outputs[0]->buffer().dim[1].extent = data[1];
                    outputs[0]->buffer().dim[2].extent = data[2];
                    outputs[0]->buffer().dim[3].extent = data[3];
                    return true;
                }
            }
        }
        // Fall back to global_size: [W, H, D] → [1, D, H, W] (4D NCHW)
        // Can be INTS list or tensor BlobT
        for (int i = 0; i < extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "global_size") {
                int gx = 1, gy = 1, gz = 1;
                bool found = false;
                if (attr->list() && attr->list()->i()) {
                    auto data = attr->list()->i();
                    if (data->size() >= 1) gx = data->data()[0];
                    if (data->size() >= 2) gy = data->data()[1];
                    if (data->size() >= 3) gz = data->data()[2];
                    found = true;
                } else if (attr->tensor() && attr->tensor()->int32s()) {
                    auto data = attr->tensor()->int32s();
                    if (data->size() >= 1) gx = data->data()[0];
                    if (data->size() >= 2) gy = data->data()[1];
                    if (data->size() >= 3) gz = data->data()[2];
                    found = true;
                }
                if (found) {
                    outputs[0]->buffer().dimensions = 4;
                    outputs[0]->buffer().dim[0].extent = 1;
                    outputs[0]->buffer().dim[1].extent = gz;
                    outputs[0]->buffer().dim[2].extent = gy;
                    outputs[0]->buffer().dim[3].extent = gx;
                    return true;
                }
            }
        }
        // If no shape info at all, copy from input
        if (inputs.size() > 0 && inputs[0]->buffer().dimensions >= 2) {
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

} // namespace MNN
