//
//  IspChainFusion.cpp — MNN converter optimization pass
//  Detects chains of standard MNN ops that form ISP pipeline patterns
//  and replaces them with fused VulkanFuse Extra ops (custom SPIR-V).
//
//  Architecture:
//    ONNX (standard ai.onnx ops)
//      → MNNConvert → standard MNN ops (NC4HW4)
//      → IspChainFusion detects patterns
//      → replaces with VulkanFuse Extra ops (CHW planar, SPIR-V)
//
//  Registered patterns:
//    A. FCS + Display
//       Scale(ch_scale) + UnaryOp(POW, exp=1/2.4)
//       → Extra("isp.fcs_display")
//
//    B. EE + LDCI (future)
//       Conv(3×3 laplacian) + Pool(AVG) + BinaryOp(SUB+MUL+ADD)
//       → Extra("isp.ee_ldci")
//

#include <string>
#include <vector>
#include <cmath>
#include "PostTreatUtils.hpp"
#include "MNN_generated.h"

using namespace MNN;

// ── SPIR-V bytecode imports ──
// Generated from shader_fcs_display_fused.spv and shader_ee_ldci_fused.spv
// Define MNN_ISP_EMBED_SPIRV to use embedded byte arrays.
// If not defined, pass will detect patterns but not insert SPIR-V.
#ifdef MNN_ISP_EMBED_SPIRV
#include "isp_spirv_embedded.h"
#endif

// ── Pattern Detection Helpers ──

static bool isLaplacianConv(const Convolution2DT* conv) {
    if (!conv || !conv->common) return false;
    auto& c = conv->common;
    if (c->kernelX != 3 || c->kernelY != 3) return false;
    if (c->strideX != 1 || c->strideY != 1) return false;
    if (c->dilateX != 1 || c->dilateY != 1) return false;
    // Check unsharp mask kernel: [[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]]
    int oc = c->outputCount;
    int ic = (int)conv->weight.size() / (oc * 9);
    if (ic < 1) return false;
    const float* w = conv->weight.data();
    const float expected[] = {0, -0.5f, 0, -0.5f, 3.0f, -0.5f, 0, -0.5f, 0};
    for (int i = 0; i < 9; i++)
        if (std::abs(w[i] - expected[i]) > 0.01f) return false;
    return true;
}

static bool isCCMConv(const Convolution2DT* conv) {
    if (!conv || !conv->common) return false;
    auto& c = conv->common;
    return c->kernelX == 1 && c->kernelY == 1 &&
           c->outputCount == 3 &&
           conv->weight.size() == 12;  // 3*4*1*1
}

static bool isGammaPow(const UnaryOpT* unary) {
    if (!unary || unary->opType != MNN::UnaryOpOperation_POW) return false;
    if (unary->tableInt8.size() == 4) {
        float exponent;
        memcpy(&exponent, unary->tableInt8.data(), 4);
        return std::abs(exponent - 1.0f/2.4f) < 0.01f;
    }
    return false;
}

// ── Fusion Pass Implementation ──

class IspChainFusion : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;

        // Scan for patterns
        for (int i = 0; i + 1 < (int)ops.size(); i++) {
            auto& op = ops[i];
            if (!op) continue;

            // ── Pattern A: FCS + Display ──
            // Scale(ch_scale) + UnaryOp(POW, 1/2.4)
            if (op->type == MNN::OpType_Scale) {
                auto* scale = op->main.AsScale();
                if (!scale) continue;

                // Look ahead for POW
                auto& next = ops[i+1];
                if (!next || next->type != MNN::OpType_UnaryOp) continue;
                auto* unary = next->main.AsUnaryOp();
                if (!unary || !isGammaPow(unary)) continue;

                MNN_PRINT("[IspFusion] Pattern A: FCS+Display at op %d\n", i);

                // Get dimensions from tensor shapes
                int W = 1920, H = 1080;  // default FHD

                // Calculate FCS strength from Scale
                float fcs_str = 0.0f;
                for (auto s : scale->scaleData) fcs_str += s;
                fcs_str /= std::max(1, (int)scale->scaleData.size());

                // Build fused Extra op
                op->type = MNN::OpType_Extra;
                op->main.type = MNN::OpParameter_Extra;
                auto extra = new ExtraT;
                extra->type = "isp.fcs_display";

                // ── Attributes ──
                auto add = [&](const std::string& k, auto&& fn) {
                    auto a = std::make_unique<AttributeT>();
                    a->key = k;
                    fn(a.get());
                    extra->attr.push_back(std::move(a));
                };

                add("output_shape", [&](AttributeT* a) {
                    a->tensor.reset(new BlobT);
                    a->tensor->dataType = MNN::DataType_DT_INT32;
                    a->tensor->int32s = {1, 3, H, W};
                });
                add("global_size", [&](AttributeT* a) {
                    a->tensor.reset(new BlobT);
                    a->tensor->dataType = MNN::DataType_DT_INT32;
                    a->tensor->int32s = {W, H, 1};
                });
                add("group_size", [&](AttributeT* a) {
                    a->tensor.reset(new BlobT);
                    a->tensor->dataType = MNN::DataType_DT_INT32;
                    a->tensor->int32s = {16, 16, 1};
                });
                add("optimized_dispatch", [&](AttributeT* a) { a->b = true; });
                add("const", [&](AttributeT* a) {
                    a->i = 0;
                    a->tensor.reset(new BlobT);
                    a->tensor->dataType = MNN::DataType_DT_FLOAT;
                    a->tensor->float32s = {
                        float(W), float(H),
                        fcs_str, 0.0f,      // fcs strength, offset
                        2.2f, 0.0f,          // gamma, brightness
                        0.0f, 0.0f, 0.0f     // pad[3]
                    };
                    a->b = false;  // SSBO
                });
                add("input", [&](AttributeT* a) {
                    a->i = 0;
                    a->list.reset(new ListValueT);
                    a->list->i = {0, 1};  // tensor_pos=0 (input), binding=1
                });
                add("input", [&](AttributeT* a) {
                    a->i = 0;
                    a->list.reset(new ListValueT);
                    a->list->i = {1, 2};  // tensor_pos=0 (output), binding=2
                });

#ifdef MNN_ISP_EMBED_SPIRV
                add("spirv", [&](AttributeT* a) {
                    a->tensor.reset(new BlobT);
                    a->tensor->dataType = MNN::DataType_DT_INT8;
                    a->tensor->int8s.assign(
                        (const int8_t*)kShaderFcsDisplayFusedSpv,
                        (const int8_t*)kShaderFcsDisplayFusedSpv +
                        kShaderFcsDisplayFusedSpvLen);
                });
#else
                // SPIR-V not embedded — model will need external SPIR-V
                MNN_PRINT("[IspFusion] WARNING: SPIR-V not embedded. "
                         "Define MNN_ISP_EMBED_SPIRV or add SPIR-V manually.\n");
#endif

                op->main.value = extra;
                // Output tensor stays at op1's output index
                op->outputIndexes[0] = op->outputIndexes[0];
                // Remove next op (mark as null for later cleanup)
                next.reset();
                changed = true;
                i++;  // skip the nulled op
                continue;
            }

            // ── Pattern B: EE + LDCI (TODO) ──
            // Conv(3×3 laplacian) + ... complex chain detection
            // For now, skip — will be added in future iteration
        }

        // Clean up null ops
        ops.erase(std::remove_if(ops.begin(), ops.end(),
                  [](const std::unique_ptr<OpT>& o) { return !o; }),
                  ops.end());

        if (changed) {
            MNN_PRINT("[IspFusion] Fusion complete: %zu ops remaining\n",
                      ops.size());
        }
        return true;
    }
};

// ── Static Registration ──
// Registers this pass so it's found by optimizeNet() via "IspChainFusion"
static PostConverterRegister<IspChainFusion> __isp_fusion("IspChainFusion");
