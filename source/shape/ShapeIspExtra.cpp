//
//  ShapeIspExtra.cpp
//  MNN
//
//  Size computer for direct-emit isp.* Extra ops (VulkanFuse path).
//  OpType_Extra has no registered computer, so the default "same as input"
//  rule applies: output dtype/shape copy inputs[0]. That is correct for most
//  isp.* elementwise ops, but WRONG for isp.display_uint8_argb which packs a
//  full-res RGBA float frame into a HALF-RES uint8 ARGB8888 frame (the
//  demosaic+display fusion). Without this computer the runtime allocates a
//  FLOAT tensor of the input shape and hands Kotlin float-reinterpreted
//  garbage instead of packed uint8 pixels.
//

#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"

namespace MNN {

class IspExtraSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto extra = op->main_as_Extra();
        std::string type = (extra && extra->type()) ? extra->type()->str() : "";
        if (getenv("ISP_SHAPE_DEBUG") != nullptr) {
            auto idims = inputs.empty() ? -1 : inputs[0]->buffer().dimensions;
            MNN_PRINT("[ShapeIspExtra] %s inDims=%d in=[%d,%d,%d,%d] fmt=%d\n",
                type.c_str(), idims,
                idims > 0 ? inputs[0]->buffer().dim[0].extent : -1,
                idims > 1 ? inputs[0]->buffer().dim[1].extent : -1,
                idims > 2 ? inputs[0]->buffer().dim[2].extent : -1,
                idims > 3 ? inputs[0]->buffer().dim[3].extent : -1,
                inputs.empty() ? -1 : (int)TensorUtils::getDescribe(inputs[0])->dimensionFormat);
        }

        if (type == "isp.display_uint8_argb") {
            // Input: [1, C=3, h, w] RGB float planes (post-demosaic, quad res).
            // Output: NHWC-packed [1, h, w, 4] uint8 ARGB8888.
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims < 2) return false;
            int W = input->buffer().dim[dims - 1].extent;
            int H = input->buffer().dim[dims - 2].extent;
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = H;
            output->buffer().dim[2].extent = W;
            output->buffer().dim[3].extent = 4;
            TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
            output->buffer().type = halide_type_of<uint8_t>();
            TensorUtils::setLinearLayout(output);
            if (getenv("ISP_SHAPE_DEBUG") != nullptr) {
                MNN_PRINT("[ShapeIspExtra] isp.display_uint8_argb out=[%d,%d,%d,%d]\n",
                    output->buffer().dim[0].extent, output->buffer().dim[1].extent,
                    output->buffer().dim[2].extent, output->buffer().dim[3].extent);
            }
            return true;
        }
        if (type == "isp.cfa") {
            // Packed-Bayer wire: input is NHWC [1, h, w, 4] where each pixel
            // carries its full 2x2 RGGB quad in the channel slots (h/w are
            // QUAD dims — no spatial downsample happens here). Output:
            // NCHW-tagged planes [1, 4, h, w]: plane p = channel p of every px.
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims != 4) return false;
            int w = input->buffer().dim[2].extent;   // NHWC: dim1=h, dim2=w
            int h = input->buffer().dim[1].extent;
            if (w <= 0 || h <= 0) {
                // Fallback for NCHW-tagged inputs (legacy): planes keep
                // input H/W and channels move to dim[1].
                h = input->buffer().dim[dims - 2].extent;
                w = input->buffer().dim[dims - 1].extent;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = 4;
            output->buffer().dim[2].extent = h;
            output->buffer().dim[3].extent = w;
            // The cfa SHADER writes quad-interleaved pixels (packed wire);
            // inherit the input's format tag so GPU2HOST and ConvertTensor
            // treat the buffer consistently with how it was written.
            TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
            output->buffer().type = inputs[0]->buffer().type;
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.debayer_g2") {
            // Input: 4-plane quad [N,4,h,w] (any layout tag; we address by
            // position: dim0=batch, dim1=planes, dim2=h, dim3=w).
            // Output: RGB planes [N,3,h,w] with G=(G1+G2)/2.
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims != 4) return false;
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = input->buffer().dim[0].extent;
            output->buffer().dim[1].extent = 3;
            output->buffer().dim[2].extent = input->buffer().dim[2].extent;
            output->buffer().dim[3].extent = input->buffer().dim[3].extent;
            // Same as cfa: shader writes pixel-packed RGB; inherit tag.
            TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
            output->buffer().type = inputs[0]->buffer().type;
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.demosaic_g2_ccm") {
            // Fused [debayer_g2 + CCM + x255 saturate + Cast u8].
            // Input: 4-plane quad [N,4,h,w] (+ optional runtime CCM
            // tensor as second input — shape comes from input[0] only).
            // Output: uint8 RGB planes [N,3,h,w]. dtype switches to
            // UINT8 to match the shader's byte-plane stores; layout tag
            // inherits the input like debayer_g2 does.
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims != 4) return false;
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = input->buffer().dim[0].extent;
            output->buffer().dim[1].extent = 3;
            output->buffer().dim[2].extent = input->buffer().dim[2].extent;
            output->buffer().dim[3].extent = input->buffer().dim[3].extent;
            TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
            outputs[0]->setType(DataType_DT_UINT8);
            TensorUtils::setLinearLayout(output);
            return true;
        }

        // All other isp.* ops are pixel-wise: same shape/dtype as input.
        if (inputs.size() >= 1) {
            TensorUtils::copyShape(inputs[0], outputs[0], true);
            outputs[0]->buffer().type = inputs[0]->buffer().type;
            return true;
        }
        return false;
    }
};

REGISTER_SHAPE(IspExtraSizeComputer, OpType_Extra);

} // namespace MNN
