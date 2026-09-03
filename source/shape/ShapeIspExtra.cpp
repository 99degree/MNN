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
            // Input: post-demosaic RGB planes. Unified NCHW on the wire:
            // input is [1, C, H, W] — read H = dim[2], W = dim[3] directly.
            // Output: NCHW [1, 4, H, W] uint8 ARGB8888.
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims < 2) return false;
            int H, W;
            if (dims >= 4) {
                H = input->buffer().dim[2].extent;
                W = input->buffer().dim[3].extent;
            } else if (dims == 3) {
                H = input->buffer().dim[1].extent;
                W = input->buffer().dim[2].extent;
            } else {
                W = input->buffer().dim[1].extent;
                H = input->buffer().dim[0].extent;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = 4;
            output->buffer().dim[2].extent = H;
            output->buffer().dim[3].extent = W;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = halide_type_of<uint8_t>();
            TensorUtils::setLinearLayout(output);
            if (getenv("ISP_SHAPE_DEBUG") != nullptr) {
                MNN_PRINT("[ShapeIspExtra] isp.display_uint8_argb inDims=%d in=[%d,%d,%d,%d] fmt=%d\n",
                    dims, input->buffer().dim[0].extent, input->buffer().dim[1].extent,
                    input->buffer().dim[2].extent, input->buffer().dim[3].extent,
                    TensorUtils::getDescribe(input)->dimensionFormat);
                MNN_PRINT("[ShapeIspExtra] isp.display_uint8_argb out=[%d,%d,%d,%d]\n",
                    output->buffer().dim[0].extent, output->buffer().dim[1].extent,
                    output->buffer().dim[2].extent, output->buffer().dim[3].extent);
            }
            return true;
        }
        if (type == "isp.cfa") {
            // Packed-Bayer wire on unified NCHW: input is [1, 4, h, w] where
            // the channel slots carry the 2x2 RGGB quad (h/w = QUAD dims).
            // Output: NCHW [1, 4, h, w] (channel plane layout tag).
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims != 4) return false;
            int h = input->buffer().dim[2].extent;
            int w = input->buffer().dim[3].extent;
            if (w <= 0 || h <= 0) {
                h = input->buffer().dim[dims - 2].extent;
                w = input->buffer().dim[dims - 1].extent;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = 4;
            output->buffer().dim[2].extent = h;
            output->buffer().dim[3].extent = w;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
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
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
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
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            outputs[0]->setType(DataType_DT_UINT8);
            TensorUtils::setLinearLayout(output);
            return true;
        }

        if (type == "isp.crop") {
            // isp.crop reshapes/translates a tile. Output spatial dims come
            // from the crop_params const attr (a 1×4 blob: [cropH, cropW] at
            // minimum). Unified NCHW on the wire: [1, 4, H, W].
            if (inputs.size() >= 1) {
                auto input  = inputs[0];
                auto output = outputs[0];
                int dims = input->buffer().dimensions;
                int H = 0, W = 0;
                if (dims >= 4) {
                    H = input->buffer().dim[2].extent;
                    W = input->buffer().dim[3].extent;
                } else if (dims == 3) {
                    H = input->buffer().dim[1].extent;
                    W = input->buffer().dim[2].extent;
                } else if (dims >= 2) {
                    W = input->buffer().dim[dims - 1].extent;
                    H = input->buffer().dim[dims - 2].extent;
                }
                output->buffer().dimensions = 4;
                output->buffer().dim[0].extent = 1;
                output->buffer().dim[1].extent = 4;
                output->buffer().dim[2].extent = H;
                output->buffer().dim[3].extent = W;
                TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
                output->buffer().type = input->buffer().type;
                TensorUtils::setLinearLayout(output);
                if (getenv("ISP_SHAPE_DEBUG") != nullptr) {
                    MNN_PRINT("[ShapeIspExtra] isp.crop EXPLICIT copyShape outDims=%d out=[%d,%d,%d,%d]\n",
                        output->buffer().dimensions,
                        output->buffer().dim[0].extent,
                        output->buffer().dim[1].extent,
                        output->buffer().dim[2].extent,
                        output->buffer().dim[3].extent);
                }
                return true;
            }
            return false;
        }

        // ALL isp.* ops use unified NCHW on the wire: [1, C, H, W].
        // H = dim[2], W = dim[3] for any 4-D tensor; for rank-3 use [1]=H, [2]=W.
        if (inputs.size() >= 1) {
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            int H = 0, W = 0;
            if (dims >= 4) {
                H = input->buffer().dim[2].extent;
                W = input->buffer().dim[3].extent;
            } else if (dims == 3) {
                H = input->buffer().dim[1].extent;
                W = input->buffer().dim[2].extent;
            } else if (dims == 2) {
                H = input->buffer().dim[0].extent;
                W = input->buffer().dim[1].extent;
            } else {
                TensorUtils::copyShape(input, output, true);
                output->buffer().type = input->buffer().type;
                return true;
            }
            int C = (dims >= 4) ? input->buffer().dim[1].extent : 1;
            if (C < 1) C = 1;
            if (H < 1 || W < 1) {
                TensorUtils::copyShape(input, output, true);
                output->buffer().type = input->buffer().type;
                return true;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = C;
            output->buffer().dim[2].extent = H;
            output->buffer().dim[3].extent = W;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = input->buffer().type;
            TensorUtils::setLinearLayout(output);
            if (getenv("ISP_SHAPE_DEBUG") != nullptr) {
                MNN_PRINT("[ShapeIspExtra] %s inDims=%d in=[%d,%d,%d,%d] fmt=%d -> out=[%d,%d,%d,%d] NCHW\n",
                    type.c_str(), dims,
                    input->buffer().dim[0].extent, input->buffer().dim[1].extent,
                    input->buffer().dim[2].extent, input->buffer().dim[3].extent,
                    (int)TensorUtils::getDescribe(input)->dimensionFormat,
                    output->buffer().dim[0].extent, output->buffer().dim[1].extent,
                    output->buffer().dim[2].extent, output->buffer().dim[3].extent);
            }
            return true;
        }
        return false;
    }
};

REGISTER_SHAPE(IspExtraSizeComputer, OpType_Extra);

} // namespace MNN
