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
        if (type.rfind("isp.display_uint32_", 0) == 0) {
            // isp.display_uint32_{rgba,bgra,argb}: RGB float planes [1,3,H,W] →
            // packed NCHW [1,1,H,W] uint32 (one RGBA pixel per uint32 lane).
            // Without this rule the generic fallback copies the input shape
            // [1,3,H,W] float, so the shader writes only 1/3 of the tensor and
            // the readback interprets the rest as garbage (black/snow screen).
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
            output->buffer().dim[1].extent = 1;
            output->buffer().dim[2].extent = H;
            output->buffer().dim[3].extent = W;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = halide_type_of<uint32_t>();
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.display_uint8_rgba") {
            // [FIX 2026-10-20] The rgba variant's shader writes PLANAR FLOAT RGB
            // [1,3,H,W] (0-255 display domain) — the runtime tensor is float
            // regardless of the declared elemType, and the packed-uint8 shape
            // ([1,1,4H,4W] uint8) mismatched the shader's float writes, which
            // produced the 'blue snow' / '4 horizontal bars' garbage. Pass the
            // input geometry through as float [1,3,H,W].
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
            output->buffer().dim[1].extent = 3;
            output->buffer().dim[2].extent = H;
            output->buffer().dim[3].extent = W;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = halide_type_of<float>();
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type.rfind("isp.display_uint8_", 0) == 0 && type != "isp.display_uint8_argb") {
            // isp.display_uint8_{rgba,bgra}: RGB float planes [1,3,H,W] →
            // packed NCHW [1,1,H*4,W*4] uint8. The shader writes one uint32 per
            // pixel at index y*W+x (LE bytes [R,G,B,A]), so the byte layout is
            // 4H rows × 4W bytes — NCHW [1,1,4H,4W] uint8. The old [1,4,H,W]
            // shape misdeclared the row stride (W bytes vs 4W bytes), corrupting
            // the AHB row walk.
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
            output->buffer().dim[1].extent = 1;
            output->buffer().dim[2].extent = H * 4;
            output->buffer().dim[3].extent = W * 4;
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            output->buffer().type = halide_type_of<uint8_t>();
            TensorUtils::setLinearLayout(output);
            return true;
        }
        if (type == "isp.cfa") {
            // RAW quad site-select on the unified NCHW wire:
            //   input  = [1, 1, H_src, W_src] full-res planar raw plane
            //   output = [1, 4, H_src/2, W_src/2] planar quad
            // The old rule assumed a packed [1,4,h,w] quad INPUT and copied
            // h,w through; with the planar full-res input that left the quad
            // at full resolution (4x the pixels the shader writes).
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
            output->buffer().dim[2].extent = h > 0 ? h / 2 : h;
            output->buffer().dim[3].extent = w > 0 ? w / 2 : w;
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
            // The graph input is declared dynamic ([1,1,-1,-1]) and MNN calls
            // this computer BEFORE the real dims are known for some pipelines.
            // Keep the tensor 4-channel even then (H/W stay dynamic) — copying
            // the raw shape would hand downstream ops a 1-channel tensor and
            // break CCM/demosaic inference.
            const int outH = (H > 0) ? (H / 2) : H;
            const int outW = (W > 0) ? (W / 2) : W;
            output->buffer().dimensions = 4;
            output->buffer().dim[0].extent = 1;
            output->buffer().dim[1].extent = 4;
            output->buffer().dim[2].extent = outH;
            output->buffer().dim[3].extent = outW;
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
        // ── Scalar sibling ops ───────────────────────────────────────────────
        // These consume a reduced stats/means tensor (or a runtime scalar) and
        // emit small metadata tensors. The generic tail computer would copy the
        // INPUT's shape onto every output, which is wrong here: a [9] stats
        // input would make each scalar output [9] and the readback would hand
        // Kotlin 9 elements where the graph declared 1. Declare the real
        // per-output extents instead, and only fall back to a scalar when an
        // extent is unknown.
        if (type == "isp.algo_ae" || type == "isp.algo_cct" ||
            type == "isp.algo_awb" || type == "isp.algo_gamma_scalar") {
            // output extents, in declaration order, matching the ONNX
            // value-infos the blocks emit.
            static const int kAeExtents[]     = {1, 1, 1};  // gain, lum_mean, target
            static const int kCctExtents[]    = {1};
            static const int kAwbExtents[]    = {3};        // R,G,B gains
            static const int kGammaExtents[]  = {1};
            const int* extents = nullptr;
            int nExt = 0;
            if (type == "isp.algo_ae")                 { extents = kAeExtents;    nExt = 3; }
            else if (type == "isp.algo_cct")           { extents = kCctExtents;   nExt = 1; }
            else if (type == "isp.algo_awb")           { extents = kAwbExtents;   nExt = 1; }
            else /* isp.algo_gamma_scalar */           { extents = kGammaExtents; nExt = 1; }
            if (outputs.empty() || (int)outputs.size() < nExt) return false;
            // All these ops are hosted on a device that also sees the input on
            // binding 1, so a missing/opaque input dtype must not stop us: use
            // the input dtype when present, else float.
            halide_type_t dt = inputs.empty() ? halide_type_of<float>()
                                              : inputs[0]->buffer().type;
            for (int i = 0; i < nExt; ++i) {
                auto output = outputs[i];
                output->buffer().dimensions = 1;
                output->buffer().dim[0].extent = extents[i];
                TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
                output->buffer().type = dt;
                TensorUtils::setLinearLayout(output);
            }
            return true;
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
            ispConstFloat(extra, 3, &fH);   // cropH (<=0 = full source height)
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
