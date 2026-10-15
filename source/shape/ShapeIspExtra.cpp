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

// Reads the `const` attribute the ISP emitter bakes as an ONNX FLOATS blob
// (VulkanFuse consumes the same attribute as binding 0). Returns false when the
// attribute or index is absent so callers can fall back to dims-derived values.
static bool ispConstFloat(const MNN::Extra* extra, int index, float* out) {
    if (extra == nullptr || extra->attr() == nullptr || out == nullptr) return false;
    for (int i = 0; i < extra->attr()->size(); ++i) {
        auto* a = extra->attr()->GetAs<Attribute>(i);
        if (a == nullptr || a->key() == nullptr) continue;
        if (a->key()->str() != "const") continue;
        if (a->tensor() && a->tensor()->float32s() && a->tensor()->float32s()->size() > (unsigned)index) {
            *out = a->tensor()->float32s()->data()[index];
            return true;
        }
    }
    return false;
}

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
        if (type == "isp.demosaic_g2") {
            // Same as debayer_g2 for shape inference
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
            output->buffer().type = inputs[0]->buffer().type;
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.demosaic_edge") {
            // isp.demosaic_edge: 4-channel CFA quad [N,4,H,W] -> 3-channel RGB [N,3,H,W]
            // Spatial dimensions preserved, channel count 4->3
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
            output->buffer().type = inputs[0]->buffer().type;
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.demosaic_ccm") {
            // isp.demosaic_ccm: fused demosaic + CCM, same shape as demosaic_edge
            // 4-channel CFA quad [N,4,H,W] -> 3-channel RGB [N,3,H,W]
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
            output->buffer().type = inputs[0]->buffer().type;
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.downscale") {
            // isp.downscale: NCHW RGB [1,3,H,W] -> [1,3,outH,outW].
            // The ONLY shape-changing isp op that had no computer, so the generic
            // fallback declared a FULL-RES output while the shader writes only
            // outW*outH pixels. Later ops then allocated/encoded against the wrong
            // extents and Session::resize never completed (the observed
            // tryResizeSession hang / COMPUTE_SIZE_ERROR).
            //
            // Out dims come from the `downscale_out` int32[2] attr the emitter
            // bakes; fallback is half the input (the 2x average the shader does).
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims < 4) return false;
            int inW = input->buffer().dim[dims - 1].extent;
            int inH = input->buffer().dim[dims - 2].extent;
            int outW = 0, outH = 0;
            if (extra != nullptr && extra->attr() != nullptr) {
                for (int i = 0; i < extra->attr()->size(); ++i) {
                    auto* a = extra->attr()->GetAs<Attribute>(i);
                    if (a == nullptr || a->key() == nullptr) continue;
                    if (a->key()->str() == "downscale_out") {
                        // ONNX INTS attribute; the converter stores ints in
                        // Attribute.list->i (and int64 tensors in tensor->int32s).
                        if (a->list() && a->list()->i() && a->list()->i()->size() >= 2) {
                            auto d = a->list()->i();
                            outW = d->data()[0]; outH = d->data()[1];
                        } else if (a->tensor() && a->tensor()->int32s() && a->tensor()->int32s()->size() >= 2) {
                            auto d = a->tensor()->int32s();
                            outW = d->data()[0]; outH = d->data()[1];
                        }
                    }
                }
            }
            if (outW <= 0 || outH <= 0) { outW = inW > 0 ? inW / 2 : 0; outH = inH > 0 ? inH / 2 : 0; }
            if (outW <= 0 || outH <= 0) {
                TensorUtils::copyShape(input, output, true);
                output->buffer().type = input->buffer().type;
                return true;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = input->buffer().dim[1].extent;
            output->buffer().dim[2].extent = outH;
            output->buffer().dim[3].extent = outW;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = input->buffer().type;
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.crop_blc_norm_cfa") {
            // Fused Crop+RawBLC+Normalize+CFA unpack: raw [1,1,H,W] ->
            // Bayer quad [1,4,H/2,W/2]. Without this rule the generic fallback
            // kept the RAW shape for a 4-channel half-res tensor, so every
            // downstream op inferred from the wrong extents.
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims < 4) return false;
            int H = input->buffer().dim[dims - 2].extent;
            int W = input->buffer().dim[dims - 1].extent;
            if (H < 1 || W < 1) {
                TensorUtils::copyShape(input, output, true);
                output->buffer().type = input->buffer().type;
                return true;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = 4;
            output->buffer().dim[2].extent = H / 2;
            output->buffer().dim[3].extent = W / 2;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = halide_type_of<float>();
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.lsc") {
            // isp.lsc: lens shading correction, same spatial dims, same channels
            // as input (multiplies per-pixel gain)
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims < 4) return false;
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = input->buffer().dim[0].extent;
            output->buffer().dim[1].extent = input->buffer().dim[1].extent;
            output->buffer().dim[2].extent = input->buffer().dim[2].extent;
            output->buffer().dim[3].extent = input->buffer().dim[3].extent;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = inputs[0]->buffer().type;
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.algo_gamma") {
            // isp.algo_gamma: gamma computation from analog_gain + scene_change
            // Inputs: scalar analog_gain [1], scalar scene_change [1]
            // Output: scalar gamma [1]
            if (inputs.size() >= 2) {
                auto input  = inputs[0];  // analog_gain
                auto output = outputs[0];
                int dims = input->buffer().dimensions;
                if (dims != 1 || input->buffer().dim[0].extent != 1) return false;
                output->buffer().dimensions = 1;
                output->buffer().dim[0].extent = 1;
                TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
                output->buffer().type = input->buffer().type;
                TensorUtils::setLinearLayout(output);
                return true;
            }
            return false;
        }
        if (type == "isp.lsc_min" || type == "isp.vignetting_min" ||
            type == "isp.chromatic_aberration_min") {
            // Grid-generating MINIMAL ops: output spatial dims match input.
            // (isp.raw_blc_min is NOT here — its shader is 3-channel-packed, so
            // it has its own rule below; the input-dims passthrough would
            // declare the wrong channel count.)
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims < 4) return false;
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = input->buffer().dim[0].extent;
            output->buffer().dim[1].extent = input->buffer().dim[1].extent;
            output->buffer().dim[2].extent = input->buffer().dim[2].extent;
            output->buffer().dim[3].extent = input->buffer().dim[3].extent;
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

        if (type == "isp.crop" || type == "isp.crop_blc") {
            // isp.crop / isp.crop_blc cut a tile out of the frame. The emitter
            // bakes the shader's binding-0 params as the `const` attr
            // float[7] = {W_src, H_src, cropW, cropH, startW, startH, gain}.
            // The generic fallback used to pass the INPUT dims through
            // unchanged, so every downstream op inferred the pre-crop extents.
            // Read cropW/cropH (and C) from that attr; fall back to the input.
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            int inH = 0, inW = 0, inC = 1;
            if (dims >= 4) {
                inC = input->buffer().dim[1].extent;
                inH = input->buffer().dim[dims - 2].extent;
                inW = input->buffer().dim[dims - 1].extent;
            } else if (dims == 3) {
                inC = input->buffer().dim[0].extent;
                inH = input->buffer().dim[1].extent;
                inW = input->buffer().dim[2].extent;
            } else if (dims >= 2) {
                inH = input->buffer().dim[dims - 2].extent;
                inW = input->buffer().dim[dims - 1].extent;
            } else {
                TensorUtils::copyShape(input, output, true);
                output->buffer().type = input->buffer().type;
                return true;
            }
            if (inC < 1) inC = 1;
            float fW = 0.f, fH = 0.f;
            ispConstFloat(extra, 2, &fW);   // cropW
            ispConstFloat(extra, 3, &fH);   // cropH
            int outW = (fW > 0.f) ? (int)fW : inW;
            int outH = (fH > 0.f) ? (int)fH : inH;
            if (outW <= 0 || outH <= 0) {
                TensorUtils::copyShape(input, output, true);
                output->buffer().type = input->buffer().type;
                return true;
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = inC;
            output->buffer().dim[2].extent = outH;
            output->buffer().dim[3].extent = outW;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = input->buffer().type;
            TensorUtils::setLinearLayout(output);
            if (getenv("ISP_SHAPE_DEBUG") != nullptr) {
                MNN_PRINT("[ShapeIspExtra] %s in=[%d,%d,%d,%d] const(cropW,cropH)=(%.0f,%.0f) -> out=[1,%d,%d,%d]\n",
                    type.c_str(), dims > 0 ? input->buffer().dim[0].extent : 0,
                    dims > 1 ? input->buffer().dim[1].extent : 0, inH, inW,
                    fW, fH, inC, outH, outW);
            }
            return true;
        }
        if (type == "isp.raw_blc" || type == "isp.raw_blc_min") {
            // Raw black-level correction. The SHIPPED shader is written for a
            // 3-channel packed RGB plane (`idx = (y*w + x) * 3`), so MNN must
            // declare C = 3 regardless of what the graph's RAW input declared —
            // otherwise the tensor is allocated C=1/C=4 wide while the shader
            // strides over 3 floats per pixel (and writes past the buffer).
            auto input  = inputs[0];
            auto output = outputs[0];
            int dims = input->buffer().dimensions;
            if (dims < 2) return false;
            int H = (dims >= 4) ? input->buffer().dim[2].extent : input->buffer().dim[0].extent;
            int W = (dims >= 4) ? input->buffer().dim[3].extent : input->buffer().dim[dims - 1].extent;
            if (getenv("ISP_SHAPE_DEBUG") != nullptr) {
                MNN_PRINT("[ShapeIspExtra] %s inDims=%d H=%d W=%d -> [1,3,%d,%d]\n",
                    type.c_str(), dims, H, W, H, W);
            }
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = 3;
            output->buffer().dim[2].extent = H;
            output->buffer().dim[3].extent = W;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = halide_type_of<float>();
            TensorUtils::setLinearLayout(output);
            return true;
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
