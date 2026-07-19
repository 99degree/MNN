//
//  IspChainFusion.cpp — MNN converter optimization pass (2-pass architecture)
//
//  Architecture — two passes, each running iteratively to fixpoint:
//
//    ONNX model (standard ai.onnx ops)
//      ↓ MNNConvert
//    Standard MNN ops
//      ↓ Pass 1: MICRO-FUSION — fuse ops within each ISP block's ONNX pattern
//    Logical ISP stages (one Extra op per block)
//      ↓ Pass 2: MACRO-FUSION — fuse adjacent ISP blocks into single dispatch
//    Fused ISP pipeline (3–5 GPU dispatches)
//      ↓ 3–5× faster on GPU
//
//  Pass 1 rules — MICRO-FUSION (longest chain first, single-block patterns):
//    ┌──────────────────────────────────────────────────────────────┐
//    │ Unpack block patterns:                                      │
//    │  R1c. Mod+Cast+Div+Stack+Conv(1×2) → isp.unpack_packed     │
//    │  R1.  Cast[+Div]+Conv(2×2,stride=2) → isp.unpack_blc       │
//    │  R1b. Concat+Conv(1×2)              → isp.unpack_blc       │
//    │ LDCI block patterns:                                        │
//    │  R5.  Pool+Sub+Mul+Add+Clip          → isp.ldci            │
//    │  R5b. ReduceMean+Sub+Mul+Add+Clip    → isp.ldci (Rust)     │
//    │ Demosaic block patterns:                                    │
//    │  R2.  Conv(1×1,4→3ch)               → isp.demosaic_ccm     │
//    │  R2b. Conv(4×4,1ch→3ch)             → isp.demosaic_interp  │
//    │ EE block patterns:                                          │
//    │  R4.  Conv(3×3)                      → isp.ee              │
//    │  R4b. Conv(3×5,g=3)+Mul             → isp.ee (Rust)       │
//    │ Micro clip-absorption (2 ops):                              │
//    │  R3c. Sub+Clip                       → isp.fcs             │
//    │  R6b. Conv(1×1)+Clip                 → absorb Clip         │
//    │  R2c. Conv(1×1)+Clip                 → absorb Clip         │
//    │ Single-op patterns:                                         │
//    │  R3.  Scale                           → isp.fcs             │
//    │  R6.  Pow+Clip                        → isp.display         │
//    │  R7.  Conv(1×1,3→1ch)                → isp.grayscale       │
//    │  R7b. Conv(1×1,3→4)                  → isp.argb_convert    │
//    │  R7c. Conv(1×1,3→3)                  → isp.yuv420_convert  │
//    │  R8.  Conv(2×2,stride=2)             → isp.pyramid         │
//    └──────────────────────────────────────────────────────────────┘
//
//  Pass 2 rules — MACRO-FUSION (Extra chain → fused Extra):
//    ┌────────────────────────────────────────────────────────────┐
//    │ 10.  isp.unpack_blc + isp.demosaic_ccm                    │
//    │      → isp.unpack_demosaic                                │
//    │ 10b. isp.unpack_blc + isp.demosaic(binning)               │
//    │      → isp.unpack_demosaic                                │
//    │  8.  isp.fcs + isp.display               → isp.fcs_display│
//    │  9.  isp.ee + isp.ldci                   → isp.ee_ldci    │
//    │ 11.  isp.unpack_demosaic + isp.fcs_display                │
//    │      → isp.unpack_demosaic (fuse display gamma)           │
//    │ 12.  isp.unpack_demosaic + isp.fcs                        │
//    │      → isp.unpack_demosaic (fuse FCS)                     │
//    │ 11b. isp.unpack_demosaic + ... + display                  │
//    │      → isp.unpack_demosaic (skip cosmetic intermediates)  │
//    └────────────────────────────────────────────────────────────┘

#include <string>
#include <vector>
#include <cmath>
#include "../PostTreatUtils.hpp"
#include "MNN_generated.h"

using namespace MNN;

// ── SPIR-V bytecodes ──
#ifdef MNN_ISP_EMBED_SPIRV
#include "isp_spirv_embedded.h"
#endif

// ═══════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════

template <typename Fn>
static void addAttr(MNN::ExtraT* extra, const std::string& key, Fn init) {
    std::unique_ptr<MNN::AttributeT> a(new MNN::AttributeT);
    a->key = key;
    init(a.get());
    extra->attr.push_back(std::move(a));
}

static void addSpirv(MNN::ExtraT* extra, const char* type) {
#ifdef MNN_ISP_EMBED_SPIRV
    struct { const char* type; const unsigned char* data; int len; } map[] = {
        {"isp.unpack_blc",      g_unpack_blc_spv,      g_unpack_blc_spv_len},
        {"isp.demosaic_ccm",    g_demosaic_ccm_spv,    g_demosaic_ccm_spv_len},
        {"isp.demosaic_noscale",g_demosaic_ccm_spv,    g_demosaic_ccm_spv_len},
        {"isp.fcs",             g_fcs_spv,             g_fcs_spv_len},
        {"isp.ee",              g_ee_spv,              g_ee_spv_len},
        {"isp.ldci",            g_ldci_spv,            g_ldci_spv_len},
        {"isp.display",         g_display_spv,         g_display_spv_len},
        {"isp.fcs_display",     g_fcs_display_spv,     g_fcs_display_spv_len},
        {"isp.ee_ldci",         g_ee_ldci_spv,         g_ee_ldci_spv_len},
        {"isp.unpack_demosaic", g_unpack_demosaic_spv, g_unpack_demosaic_spv_len},
        {"isp.demosaic_interp", g_demosaic_interp_spv, g_demosaic_interp_spv_len},
        // Unified isp.demosaic opset — algorithm parameter selects SPIR-V
        {"isp.demosaic_binning", g_unpack_blc_spv,      g_unpack_blc_spv_len},
        {"isp.demosaic_bilinear", g_demosaic_interp_spv, g_demosaic_interp_spv_len},
        {"isp.demosaic_mhc",     g_demosaic_mhc_spv,     g_demosaic_mhc_spv_len},
        {"isp.grayscale",       g_grayscale_spv,       g_grayscale_spv_len},
        {"isp.argb_convert",    g_argb_convert_spv,    g_argb_convert_spv_len},
        {"isp.yuv420_convert",  g_yuv420_convert_spv,  g_yuv420_convert_spv_len},
        {"isp.unpack_packed",   g_unpack_packed_spv,   g_unpack_packed_spv_len},
        {"isp.pyramid",         g_pyramid_spv,         g_pyramid_spv_len},
        {"isp.warp",             g_warp_spv,             g_warp_spv_len},
        // New post-processing shaders
        {"isp.vignetting",       g_vignetting_spv,       g_vignetting_spv_len},
        {"isp.auto_contrast",    g_auto_contrast_spv,    g_auto_contrast_spv_len},
        {"isp.colorspace",       g_colorspace_spv,       g_colorspace_spv_len},
        {"isp.wavelet_denoise",  g_wavelet_denoise_spv,  g_wavelet_denoise_spv_len},
        {"isp.bilateral",        g_bilateral_spv,        g_bilateral_spv_len},
    };
    for (auto& m : map) {
        if (strcmp(type, m.type) == 0) {
            addAttr(extra, "spirv", [&](MNN::AttributeT* a) {
                a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_INT8;
                a->tensor->int8s.assign((const int8_t*)m.data,
                                         (const int8_t*)m.data + m.len);
            });
            return;
        }
    }
    LOG(WARNING) << "[IspFusion] No SPIR-V for '" << type << "'\n";
#else
    LOG(WARNING) << "[IspFusion] MNN_ISP_EMBED_SPIRV not defined\n";
#endif
}

// Forward declaration
static void setEngine(MNN::ExtraT* extra);

// ── Unified isp.demosaic opset helper ──
// Creates an Extra op with algorithm parameter.
// Supported: binning, bilinear, mhc, ahd
static void makeDemosaic(MNN::OpT* op, const char* algorithm,
                         const std::vector<float>& uniforms) {
    if (!op) return;
    op->type = MNN::OpType_Extra;
    op->main.type = MNN::OpParameter_Extra;
    auto* ex = new MNN::ExtraT();
    static const struct { const char* algo; const char* spv; } map[] = {
        {"binning",  "isp.demosaic_binning"},
        {"bilinear", "isp.demosaic_bilinear"},
        {"mhc",      "isp.demosaic_mhc"},
        {"warp",      "isp.warp"},
    };
    const char* spv_type = "isp.demosaic_bilinear";
    for (auto& m : map) {
        if (strcmp(algorithm, m.algo) == 0) { spv_type = m.spv; break; }
    }
    ex->type = spv_type;
    setEngine(ex);
    addAttr(ex, "algorithm", [&](MNN::AttributeT* a) { a->s = algorithm; });
    addSpirv(ex, spv_type);
    {
        std::unique_ptr<MNN::AttributeT> a(new MNN::AttributeT);
        a->key = "uniforms";
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_FLOAT;
        a->tensor->float32s = uniforms;
        ex->attr.push_back(std::move(a));
    }
    op->main.value = ex;
}

// Common Extra attributes for ISP ops
static void buildCommonAttrs(MNN::ExtraT* extra, int W, int H,
                              const std::vector<float>& uniforms) {
    addAttr(extra, "output_shape", [&](MNN::AttributeT* a) {
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_INT32;
        a->tensor->int32s = {1, 3, H, W};
    });
    addAttr(extra, "global_size", [&](MNN::AttributeT* a) {
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_INT32;
        a->tensor->int32s = {W, H, 1};
    });
    addAttr(extra, "group_size", [&](MNN::AttributeT* a) {
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_INT32;
        a->tensor->int32s = {16, 16, 1};
    });
    addAttr(extra, "optimized_dispatch", [&](MNN::AttributeT* a) { a->b = true; });
    addAttr(extra, "fp16_consts", [&](MNN::AttributeT* a) { a->b = true; });
    addAttr(extra, "input", [&](MNN::AttributeT* a) {
        a->i = 0;
        a->list.reset(new MNN::ListValueT); a->list->i = {0, 1};
    });
    addAttr(extra, "input", [&](MNN::AttributeT* a) {
        a->i = 0;
        a->list.reset(new MNN::ListValueT); a->list->i = {1, 2};
    });
    addAttr(extra, "const", [&](MNN::AttributeT* a) {
        a->i = 0;
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_FLOAT;
        a->tensor->float32s = uniforms;
        a->b = false;  // SSBO std430
    });
}

// Store named float vector as an attribute on an Extra op.
static void addNamedFloats(MNN::ExtraT* ex, const char* key,
                           const std::vector<float>& vals) {
    addAttr(ex, key, [&](MNN::AttributeT* a) {
        a->i = 0;
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_FLOAT;
        a->tensor->float32s = vals;
    });
}

// Set engine="MNN" so converter validation passes for custom Extra ops
static void setEngine(MNN::ExtraT* extra) {
    extra->engine = "MNN";
}

// Trace a tensor index through ConvertTensor ops to find the original producer
// Returns the original tensor index, or the input if no ConvertTensor found
static int traceTensor(int tensorIdx, const std::vector<std::unique_ptr<OpT>>& ops) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& op : ops) {
            if (op && op->type == MNN::OpType_ConvertTensor &&
                op->inputIndexes.size() == 1 && op->outputIndexes.size() == 1 &&
                op->outputIndexes[0] == tensorIdx) {
                tensorIdx = op->inputIndexes[0];
                changed = true;
                break;
            }
        }
    }
    return tensorIdx;
}

static bool isChain(const OpT* a, const OpT* b) {
    return a && b && !a->outputIndexes.empty() && !b->inputIndexes.empty()
           && a->outputIndexes[0] == b->inputIndexes[0];
}

// Check if two ops are chained, skipping ConvertTensors between them.
// Also checks if b's input traces back to a's output through ConvertTensors.
static bool isChainSkipCT(const OpT* a, const OpT* b,
                           const std::vector<std::unique_ptr<OpT>>& ops) {
    if (!a || !b || a->outputIndexes.empty() || b->inputIndexes.empty())
        return false;
    int aOut = a->outputIndexes[0];
    // Check each input of b
    for (int inIdx : b->inputIndexes) {
        if (traceTensor(inIdx, ops) == aOut) return true;
    }
    return false;
}

static bool isExtraOfType(const OpT* op, const char* type) {
    if (!op || op->type != MNN::OpType_Extra) return false;
    auto* e = op->main.AsExtra();
    return e && e->type == type;
}

// Helper: extract the "const" attribute's float data from an Extra op
static std::vector<float> getExtraConst(const std::unique_ptr<OpT>& op) {
    if (!op || op->type != MNN::OpType_Extra) return {};
    auto* ex = op->main.AsExtra();
    if (!ex) return {};
    for (auto& attr : ex->attr) {
        if (attr && attr->key == "const" && attr->tensor &&
            !attr->tensor->float32s.empty()) {
            return attr->tensor->float32s;
        }
    }
    return {};
}

// Read a named float vector from Extra op attributes.
// Each ISP stage stores its params under its own key (e.g. "blc", "wb", "fcs").
static std::vector<float> getNamedFloats(MNN::ExtraT* ex, const char* key) {
    if (!ex) return {};
    for (auto& attr : ex->attr) {
        if (attr && attr->key == key && attr->tensor &&
            !attr->tensor->float32s.empty()) {
            return attr->tensor->float32s;
        }
    }
    return {};
}

static bool isAvgPool3x3(const PoolT* p) {
    return p && p->type == MNN::PoolType_AVEPOOL
           && p->kernelX == 3 && p->kernelY == 3
           && p->strideX == 1 && p->strideY == 1;
}

static bool isUnpackConv(const Convolution2DT* c, bool* is3chOut) {
    if (!c || !c->common) return false;
    if (c->common->outputCount != 4 && c->common->outputCount != 3) return false;
    if (is3chOut) *is3chOut = (c->common->outputCount == 3);
    // Python pattern: 2×2 kernel, stride 2 (raw Bayer input [1,1,H,W])
    if (c->common->kernelX == 2 && c->common->kernelY == 2
        && c->common->strideX == 2 && c->common->strideY == 2)
        return true;
    // Rust pattern: 1×2 kernel, stride 1×2 (pre-processed [1,2,H,W/2] from Concat)
    if (c->common->kernelX == 1 && c->common->kernelY == 2
        && c->common->strideX == 1 && c->common->strideY == 2)
        return true;
    // Generalized unpack: kernelY=2, strideY=2, kernelX=strideX=sw
    // sw can be 1,2,4,... (width downscale factor for NativeInt16)
    if (c->common->kernelY == 2 && c->common->strideY == 2
        && c->common->kernelX == c->common->strideX
        && c->common->kernelX >= 1)
        return true;
    return false;
}

static bool isUnpackConv(const Convolution2DT* c) {
    return isUnpackConv(c, nullptr);
}

static bool isCcmConv(const Convolution2DT* c) {
    return c && c->common && c->common->kernelX == 1 && c->common->kernelY == 1
           && c->common->outputCount == 3;
}

static bool isEeConv(const Convolution2DT* c) {
    if (!c || !c->common || c->common->kernelX != 3 || c->common->kernelY != 3) return false;
    if (c->common->outputCount != 3) return false;
    const float expected[9] = {0, -0.5f, 0, -0.5f, 3.0f, -0.5f, 0, -0.5f, 0};
    int ic = (int)c->weight.size() / 27;  // 3 oc × 3 ic × 9
    if (ic < 1) return false;
    for (int i = 0; i < 9; i++)
        if (std::abs(c->weight[i] - expected[i]) > 0.01f) return false;
    return true;
}

static bool isBinaryType(const OpT* op, int bt) {
    auto* b = op ? op->main.AsBinaryOp() : nullptr;
    return b && b->opType == bt;
}

// skip Const, ConvertTensor, Reshape, Squeeze, Unsqueeze, Identity
static int skipThroughAll(int start, const std::vector<std::unique_ptr<OpT>>& ops) {
    int j = start;
    while (j < (int)ops.size() && ops[j]) {
        auto t = ops[j]->type;
        if (t == MNN::OpType_Const ||
            t == MNN::OpType_ConvertTensor ||
            t == MNN::OpType_Reshape ||
            t == MNN::OpType_Squeeze ||
            t == MNN::OpType_Unsqueeze ||
            t == MNN::OpType_Identity) {
            j++;
        } else {
            break;
        }
    }
    return j;
}

// skip backward through same set of ops
static int skipThroughAllBackward(int start, const std::vector<std::unique_ptr<OpT>>& ops) {
    int j = start;
    while (j >= 0 && ops[j]) {
        auto t = ops[j]->type;
        if (t == MNN::OpType_Const ||
            t == MNN::OpType_ConvertTensor ||
            t == MNN::OpType_Reshape ||
            t == MNN::OpType_Squeeze ||
            t == MNN::OpType_Unsqueeze ||
            t == MNN::OpType_Identity) {
            j--;
        } else {
            break;
        }
    }
    return j;
}

// Find the next non-Const, non-ConvertTensor op starting from index `start`
static int skipThrough(int start, const std::vector<std::unique_ptr<OpT>>& ops) {
    int j = start;
    while (j < (int)ops.size() && ops[j] &&
           (ops[j]->type == MNN::OpType_Const ||
            ops[j]->type == MNN::OpType_ConvertTensor)) j++;
    return j;
}

// ═══════════════════════════════════════════════════════════════════
//  Pass 1: Standard MNN ops → ISP Extra ops
// ═══════════════════════════════════════════════════════════════════

class Pass1_ToExtra : public PostConverter {
public:
    static Pass1_ToExtra* instance() {
        static Pass1_ToExtra p;
        return &p;
    }

    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;

        // Extract input dimensions from the Input op
        for (auto& op : ops) {
            if (op && op->type == MNN::OpType_Input) {
                auto* inp = op->main.AsInput();
                if (inp && inp->dims.size() >= 4) {
                    mInH = inp->dims[2];  // NCHW
                    mInW = inp->dims[3];
                }
                break;
            }
        }
        // Output dimensions after stride-2 unpack
        mH = mInH / 2;
        mW = mInW / 2;

        VLOG(2) << "[P1] Input dims: " << mInH << "x" << mInW
                << " → Output dims: " << mH << "x" << mW;

        // Pre-pass: convert ONNX Extras (Conv, AveragePool, Clip) to native MNN ops.
        // These are created by DefaultonnxOpConverter during ONNX import and would
        // only be converted by RunExtraPass (which runs AFTER IspChainFusion).
        // By converting them here, ISP detection rules can check for native op types.
        for (int i = 0; i < (int)ops.size(); i++) {
            if (!ops[i] || ops[i]->type != MNN::OpType_Extra) continue;
            if (ops[i]->main.type != MNN::OpParameter_Extra) continue;
            auto* ex = ops[i]->main.AsExtra();
            if (!ex) continue;

            if (ex->type == "Conv") {
                // Convert Extra(type="Conv") to native Convolution2D
                int kernelY = 1, kernelX = 1, strideY = 1, strideX = 1, group = 1;
                std::vector<int> pads = {0, 0, 0, 0};
                for (const auto& attr_ptr : ex->attr) {
                    if (!attr_ptr) continue;
                    auto* a = attr_ptr.get();
                    if (a->key == "kernel_shape" && a->list && a->list->i.size() >= 2) {
                        kernelY = a->list->i[0]; kernelX = a->list->i[1];
                    } else if (a->key == "strides" && a->list && a->list->i.size() >= 2) {
                        strideY = a->list->i[0]; strideX = a->list->i[1];
                    } else if (a->key == "pads" && a->list && a->list->i.size() >= 4) {
                        for (int p = 0; p < 4; p++) pads[p] = a->list->i[p];
                    } else if (a->key == "group") {
                        group = a->i;
                    }
                }
                // Find weight and bias from Const ops
                const float* weightData = nullptr;
                const float* biasData = nullptr;
                int weightSize = 0, biasSize = 0;
                int oc_dims = 0, ic_dims = 0;  // from weight tensor shape
                if (ops[i]->inputIndexes.size() >= 2) {
                    int wIn = ops[i]->inputIndexes[1];
                    int bIn = ops[i]->inputIndexes.size() >= 3 ? ops[i]->inputIndexes[2] : -1;
                    for (int j = 0; j < (int)ops.size(); j++) {
                        if (!ops[j] || ops[j]->type != MNN::OpType_Const) continue;
                        for (int outIdx : ops[j]->outputIndexes) {
                            if (outIdx == wIn && ops[j]->main.AsBlob()) {
                                auto* blb = ops[j]->main.AsBlob();
                                weightData = blb->float32s.data();
                                weightSize = (int)blb->float32s.size();
                                // Try to read original weight shape from the Const blob
                                if (blb->dims.size() >= 4) {
                                    oc_dims = blb->dims[0];  // output channels
                                    ic_dims = blb->dims[1];  // input channels per group
                                }
                            }
                            if (outIdx == bIn && ops[j]->main.AsBlob()) {
                                auto* blb = ops[j]->main.AsBlob();
                                biasData = blb->float32s.data();
                                biasSize = (int)blb->float32s.size();
                            }
                        }
                    }
                }
                if (!weightData || weightSize < 1) continue;
                // Infer oc and ic from weight shape [oc, ic_per_group, ky, kx]
                int oc = 0, ic = 1;
                // First try dims from the Const blob (set above)
                if (oc_dims > 0 && ic_dims > 0) {
                    oc = oc_dims;
                    ic = ic_dims;
                    if (oc * ic * kernelY * kernelX != weightSize) {
                        oc = 0;  // fall back to inference
                    }
                }
                if (oc == 0) {
                    if (biasSize > 0) {
                        oc = biasSize;
                        ic = weightSize / (oc * kernelY * kernelX);
                    } else {
                        ic = group;
                        oc = weightSize / (ic * kernelY * kernelX);
                    }
                    if (oc * ic * kernelY * kernelX != weightSize) {
                        ic = 1;
                        oc = weightSize / (kernelY * kernelX);
                    }
                }

                auto* convParam = new MNN::Convolution2DT;
                convParam->common.reset(new MNN::Convolution2DCommonT);
                convParam->common->kernelX = kernelX;
                convParam->common->kernelY = kernelY;
                convParam->common->strideX = strideX;
                convParam->common->strideY = strideY;
                convParam->common->padX = pads.size() >= 2 ? pads[1] : 0;
                convParam->common->padY = pads.size() >= 1 ? pads[0] : 0;
                convParam->common->group = group;
                convParam->common->inputCount = ic * group;
                convParam->common->outputCount = oc;
                convParam->common->dilateX = 1;
                convParam->common->dilateY = 1;
                convParam->weight.assign(weightData, weightData + weightSize);
                convParam->bias.assign(biasData, biasData + biasSize);

                if (group > 1 && group == oc && ic * group == oc) {
                    ops[i]->type = MNN::OpType_ConvolutionDepthwise;
                } else {
                    ops[i]->type = MNN::OpType_Convolution;
                }
                ops[i]->main.type = MNN::OpParameter_Convolution2D;
                ops[i]->main.value = convParam;
                VLOG(2) << "[P1]  ConvNative at " << i << " k=" << kernelY << "x" << kernelX
                        << " oc=" << oc << " ic=" << ic << " g=" << group;
            } else if (ex->type == "AveragePool") {
                int kernelY = 3, kernelX = 3, strideY = 1, strideX = 1;
                std::vector<int> pads = {1, 1, 1, 1};
                for (const auto& attr_ptr : ex->attr) {
                    if (!attr_ptr) continue;
                    auto* a = attr_ptr.get();
                    if (a->key == "kernel_shape" && a->list && a->list->i.size() >= 2) {
                        kernelY = a->list->i[0]; kernelX = a->list->i[1];
                    } else if (a->key == "strides" && a->list && a->list->i.size() >= 2) {
                        strideY = a->list->i[0]; strideX = a->list->i[1];
                    } else if (a->key == "pads" && a->list && a->list->i.size() >= 4) {
                        for (int p = 0; p < 4; p++) pads[p] = a->list->i[p];
                    }
                }
                auto* poolParam = new MNN::PoolT;
                poolParam->type = MNN::PoolType_AVEPOOL;
                poolParam->kernelY = kernelY;
                poolParam->kernelX = kernelX;
                poolParam->strideY = strideY;
                poolParam->strideX = strideX;
                poolParam->padY = pads[0];
                poolParam->padX = pads[1];
                poolParam->padType = MNN::PoolPadType_CAFFE;
                ops[i]->type = MNN::OpType_Pooling;
                ops[i]->main.type = MNN::OpParameter_Pool;
                ops[i]->main.value = poolParam;
                VLOG(2) << "[P1]  AvgPoolNative at " << i;
            } else if (ex->type == "Clip") {
                // Convert ONNX Clip with min=0,max=1 to native ReLU6
                auto* reluParam = new MNN::ReluT;
                reluParam->slope = 0.0f;
                ops[i]->type = MNN::OpType_ReLU6;
                ops[i]->main.type = MNN::OpParameter_Relu;
                ops[i]->main.value = reluParam;
                VLOG(2) << "[P1]  ClipNative at " << i;
            }
        }

        for (int i = 0; i < (int)ops.size(); i++) {
            if (!ops[i]) continue;

            // ═══ LONGEST CHAIN FIRST (prevents short rules from consuming ops) ═══

            // ═══════════════════════════════════════════════════════════════
            // PASS 1: MICRO-FUSION — fuse ops within each ISP block's ONNX pattern
            // Each rule matches the ONNX node pattern emitted by ONE block.
            // Longest chain first → prevents short rules from consuming ops
            // that belong to a longer block pattern.
            // ═══════════════════════════════════════════════════════════════

            // ── Unpack block patterns (longest first) ──
            // R1c: Full packed-int32 chain (~8 ops) → isp.unpack_packed
            // UnpackCfaBlock PackedInt32: Mod+Cast+Div+Cast+Div+Div+Stack+Conv
            // Skips if demosaic Conv follows → lets Pass 2 macro-fuse instead.
            if (tryUnpackPackedChain(ops, i)) { changed = true; continue; }
            // R1: Cast[+Div]+Conv(2×2,stride=2,4ch) → isp.unpack_blc
            // UnpackCfaBlock NativeInt16: Cast+Conv or Div+Cast+Conv
            if (tryUnpack(ops, i)) { changed = true; continue; }
            // R1b: Rust Concat+Conv(1×2,stride=1×2,4ch) → isp.unpack_blc
            if (tryUnpackRust(ops, i)) { changed = true; continue; }

            // ── LDCI block patterns (~5 ops) ──
            // R5: Pool(AVG,3×3)+Sub+Mul+Add+ReLU6 → isp.ldci
            if (tryLdci(ops, i)) { changed = true; continue; }
            // R5b: ReduceMean+Sub+Mul+Mul+Add+ReLU6 → isp.ldci (Rust variant)
            if (tryRustReduceLdci(ops, i)) { changed = true; continue; }

            // ── Demosaic block patterns ──
            // R2: Conv(1×1,4→3ch) → isp.demosaic_ccm
            if (tryDemosaic(ops, i)) { changed = true; continue; }
            // R2b: Conv(4×4,stride=1,1ch→3ch) → isp.demosaic_interp
            if (tryDemosaicInterp(ops, i)) { changed = true; continue; }

            // ── EE block patterns ──
            // R4: Conv(3×3) → isp.ee
            if (tryEe(ops, i)) { changed = true; continue; }
            // R4b: Conv(3×5,g=3,laplacian)+Mul → isp.ee (Rust variant)
            if (tryRustConvEe(ops, i)) { changed = true; continue; }

            // ── Micro fusion FIRST (before block rules consume ops) ──
            // R6c: ReLU6 → absorb into Conv producer
            // MUST run before R2/R4 — those convert Conv to Extra, preventing absorption.
            if (tryClipAbsorbFwd(ops, i)) { changed = true; continue; }
            // R3c: Sub+ReLU6 → isp.fcs (white-level normalize)
            if (trySubClipNormalize(ops, i)) { changed = true; continue; }
            // R3d: Sub+Mul(difference scaling) → fuse
            if (trySubMul(ops, i)) { changed = true; continue; }
            // R3e: Mul+Add(channel bias) → fuse
            if (tryMulAdd(ops, i)) { changed = true; continue; }
            // R6d: Mul+ReLU6 → absorb (white-level clamp)
            if (tryMulClip(ops, i)) { changed = true; continue; }
            // R3f: Sub+Max+Min → fuse (BLC50)
            if (trySubMaxMin(ops, i)) { changed = true; continue; }
            // R4c: Conv(3×3)+Sub → fuse (unsharp sharpen)
            if (tryConvSub(ops, i)) { changed = true; continue; }

            // ── Single-block single-op patterns ──
            // R3: Scale → isp.fcs
            if (tryFcs(ops, i)) { changed = true; continue; }
            // R7b: Conv(1×1,3→4) → isp.argb_convert
            if (tryArgbConvert(ops, i)) { changed = true; continue; }
            // R7c: Conv(1×1,3→3) → isp.yuv420_convert
            if (tryYuv420Convert(ops, i)) { changed = true; continue; }
            // R7: Conv(1×1,3→1ch) → isp.grayscale
            if (tryGrayscale(ops, i)) { changed = true; continue; }
            // R7d: Log+Mul+Exp → isp.gamma (GammaBlock)
            // if (tryGamma(ops, i)) { changed = true; continue; }
            // R6: Pow+ReLU6 → isp.display
            if (tryDisplay(ops, i)) { changed = true; continue; }
            // R8: Conv(2×2,stride=2) → isp.pyramid
            if (tryPyramid(ops, i)) { changed = true; continue; }
            // Rwarp: Extra(isp.warp)
            if (tryWarp(ops, i)) { changed = true; continue; }
            // Rust variants: FCS, ExtraEe, Display
            if (tryRustConvFcs(ops, i)) { changed = true; continue; }
            if (tryRustExtraEe(ops, i)) { changed = true; continue; }
            if (tryRustDisplay(ops, i)) { changed = true; continue; }
            // New post-processing ops
            // R13: Mul(input, gain_map) → isp.vignetting
            if (tryVignetting(ops, i)) { changed = true; continue; }
            // R14: Add → Sub → Mul → Add chain → isp.auto_contrast
            if (tryAutoContrast(ops, i)) { changed = true; continue; }
            // NEW: Missing ISP blocks
            // DPC (median filter) - 3x3 Pool -> Sub -> Mul -> Add -> Clip
            if (tryDpc(ops, i)) { changed = true; continue; }
            // Gaussian Denoise - Conv -> Add (blend)
            if (tryGaussianDenoise(ops, i)) { changed = true; continue; }
            // LSC (Lens Shading Correction) - Mul with 2D gain map
            if (tryLsc(ops, i)) { changed = true; continue; }
            // AWB (Auto White Balance) - Mul(3ch gains) + Add(3ch offsets)
            if (tryAwb(ops, i)) { changed = true; continue; }
            // AE (Auto Exposure) - Mul(global gain) + optional Add(offset)
            if (tryAe(ops, i)) { changed = true; continue; }
            // Tone Mapping - Pow(gamma) -> Mul(contrast) -> optional unsharp
            if (tryTone(ops, i)) { changed = true; continue; }
            // Gamma - Pow operation
            if (tryGamma(ops, i)) { changed = true; continue; }
            // Calibration Stats - ReduceMean/Min/Max
            if (tryCalibStats(ops, i)) { changed = true; continue; }
            // IspController Stats - ReduceMean for AE/AWB/AF
            if (tryIspControllerStats(ops, i)) { changed = true; continue; }
            // AF Focus - Sobel -> Mul -> ReduceMean -> Pow -> ReduceMean
            if (tryAfFocus(ops, i)) { changed = true; continue; }
            // EIS Gyro - warp with gyro params
            if (tryEisGyro(ops, i)) { changed = true; continue; }

            // NEW: Missing ISP blocks
            // DPC (median filter) - 3x3 Pool -> Sub -> Mul -> Add -> Clip
            if (tryDpc(ops, i)) { changed = true; continue; }
            // Gaussian Denoise - Conv -> Add (blend)
            if (tryGaussianDenoise(ops, i)) { changed = true; continue; }
            // LSC (Lens Shading Correction) - Mul with 2D gain map
            if (tryLsc(ops, i)) { changed = true; continue; }
            // AWB (Auto White Balance) - Mul(3ch gains) + Add(3ch offsets)
            if (tryAwb(ops, i)) { changed = true; continue; }
            // AE (Auto Exposure) - Mul(global gain) + optional Add(offset)
            if (tryAe(ops, i)) { changed = true; continue; }
            // Tone Mapping - Pow(gamma) -> Mul(contrast) -> optional unsharp
            if (tryTone(ops, i)) { changed = true; continue; }
            // Gamma - Pow operation
            if (tryGamma(ops, i)) { changed = true; continue; }
            // Calibration Stats - ReduceMean/Min/Max
            if (tryCalibStats(ops, i)) { changed = true; continue; }
            // IspController Stats - ReduceMean for AE/AWB/AF
            if (tryIspControllerStats(ops, i)) { changed = true; continue; }
            // AF Focus - Sobel -> Mul -> ReduceMean -> Pow -> ReduceMean
            if (tryAfFocus(ops, i)) { changed = true; continue; }
            // EIS Gyro - warp with gyro params
            if (tryEisGyro(ops, i)) { changed = true; continue; }

        }

        ops.erase(std::remove_if(ops.begin(), ops.end(),
                  [](const std::unique_ptr<OpT>& o) { return !o; }), ops.end());
        return changed;
    }

private:
    // Dimensions inferred from model input shape at onExecute time
    mutable int mW = 1920, mH = 1080;       // output (after stride-2)
    mutable int mInW = 3840, mInH = 2160;   // input (Bayer raw)

    // R1: Cast [+Div(Normalize)] + Conv(2×2,stride=2,3-4ch) → isp.unpack_blc or isp.demosaic(binning)
    // Enhanced: also absorps Normalize (Div÷max) before Cast, and BLC (Sub+Clip) after Conv.
    bool tryUnpack(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        // Scan backward from i to find a Cast feeding into this Conv
        int castIdx = -1;
        if (ops[i]->type == MNN::OpType_Cast) {
            castIdx = i;  // Cast IS at current position
        } else {
            // Cast may be just before current position. Scan backward.
            int prev = skipThroughAllBackward(i - 1, ops);
            if (prev >= 0 && ops[prev] && ops[prev]->type == MNN::OpType_Cast) {
                castIdx = prev;
            }
        }
        
        // Scan backward from castIdx to find Normalize (Div with const) before Cast
        float normScale = 1023.0f;  // default sensor_max
        int divIdx = -1;
        if (castIdx >= 0) {
            // Check if there's a BinaryOp(DIV) feeding into the Cast
            for (int inIdx : ops[castIdx]->inputIndexes) {
                for (int j = 0; j < (int)ops.size(); j++) {
                    if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
                    if (!isBinaryType(ops[j].get(), MNN::BinaryOpOperation_DIV)) continue;
                    for (int outIdx : ops[j]->outputIndexes) {
                        if (outIdx == inIdx) {
                            // Found Div preceding Cast — check if second input is const
                            for (int divInIdx : ops[j]->inputIndexes) {
                                if (divInIdx != inIdx) {  // not the first input
                                    // Look for Const tensor
                                    for (int k = 0; k < (int)ops.size(); k++) {
                                        if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                                        for (int constOut : ops[k]->outputIndexes) {
                                            if (constOut == divInIdx) {
                                                auto* blb = ops[k]->main.AsBlob();
                                                if (blb && !blb->float32s.empty()) {
                                                    float maxVal = blb->float32s[0];
                                                    if (maxVal > 0.0f) {
                                                        normScale = maxVal;
                                                        divIdx = j;
                                                    }
                                                }
                                                break;
                                            }
                                        }
                                        if (divIdx >= 0) break;
                                    }
                                }
                                if (divIdx >= 0) break;
                            }
                            break;
                        }
                    }
                    if (divIdx >= 0) break;
                }
                if (divIdx >= 0) break;
            }
        }

        // Find the Conv — it may not be immediately after Cast (Div may be between)
        int ci = castIdx >= 0 ? skipThroughAll(castIdx + 1, ops) : i;
        if (ci >= (int)ops.size() || !ops[ci]) return false;
        if (ops[ci]->type != MNN::OpType_Convolution) return false;
        auto* conv = ops[ci]->main.AsConvolution2D();
        bool is3chOut = false;
        if (!isUnpackConv(conv, &is3chOut)) return false;

        // Scan forward after Conv for BLC (Sub + ReLU6/Clip)
        float blc0 = 0.0f, blc1 = 0.0f, blc2 = 0.0f, blc3 = 0.0f;
        int blcSubIdx = -1, blcClipIdx = -1;
        int afterConv = skipThroughAll(ci + 1, ops);
        if (afterConv < (int)ops.size() && ops[afterConv] &&
            ops[afterConv]->type == MNN::OpType_BinaryOp &&
            isBinaryType(ops[afterConv].get(), MNN::BinaryOpOperation_SUB)) {
            // Found Sub — extract blc_vals
            for (int subInIdx : ops[afterConv]->inputIndexes) {
                // Look for Const in the second input
                if (subInIdx != ops[afterConv]->inputIndexes[0]) {
                    // Find Const tensor
                    for (int k = 0; k < (int)ops.size(); k++) {
                        if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                        for (int constOut : ops[k]->outputIndexes) {
                            if (constOut == subInIdx) {
                                auto* blb = ops[k]->main.AsBlob();
                                if (blb && blb->float32s.size() >= 4) {
                                    blc0 = blb->float32s[0];
                                    blc1 = blb->float32s[1];
                                    blc2 = blb->float32s[2];
                                    blc3 = blb->float32s[3];
                                }
                                break;
                            }
                        }
                    }
                }
            }
            blcSubIdx = afterConv;
            // Check for ReLU6/Clip after Sub
            int clipIdx = skipThroughAll(afterConv + 1, ops);
            if (clipIdx < (int)ops.size() && ops[clipIdx] &&
                (ops[clipIdx]->type == MNN::OpType_ReLU ||
                 ops[clipIdx]->type == MNN::OpType_ReLU6)) {
                blcClipIdx = clipIdx;
            }
        }

        // Build const buffer with Normalize + BLC
        std::vector<float> u = {float(mW),float(mH),float(mInW),float(mInH),
                                normScale,
                                blc0, blc1, blc2, blc3,
                                1,1,1,1};

        if (is3chOut) {
            // 1ch→3ch binning → isp.demosaic(algo=binning)
            makeDemosaic(ops[ci].get(), "binning", u);
            VLOG(2) << "[P1] R1: demosaic(binning) at " << i
                    << " scale=" << normScale
                    << " blc=[" << blc0 << "," << blc1 << "," << blc2 << "," << blc3 << "]";
        } else {
            // Compute actual output dimensions from Conv strides
            int outH = (int)(mInH / conv->common->strideY);
            int outW = (int)(mInW / conv->common->strideX);
            if (outH <= 0) outH = mH;
            if (outW <= 0) outW = mW;

            // 1ch→4ch unpack → isp.unpack_blc
            ops[ci]->type = MNN::OpType_Extra;
            ops[ci]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.unpack_blc";
            addAttr(ex, "output_shape", [&](MNN::AttributeT* a) {
                a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_INT32;
                a->tensor->int32s = {1,4,outH,outW};
            });
            addAttr(ex, "global_size", [&](MNN::AttributeT* a) {
                a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_INT32;
                a->tensor->int32s = {outW,outH,1};
            });
            addAttr(ex, "group_size", [&](MNN::AttributeT* a) {
                a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_INT32;
                a->tensor->int32s = {16,16,1};
            });
            addAttr(ex, "optimized_dispatch", [&](MNN::AttributeT* a) { a->b = true; });
            addAttr(ex, "input", [&](MNN::AttributeT* a) {
                a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {0,1};
            });
            addAttr(ex, "input", [&](MNN::AttributeT* a) {
                a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {1,2};
            });
            addAttr(ex, "const", [&](MNN::AttributeT* a) {
                a->i = 0; a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_FLOAT;
                a->tensor->float32s = u; a->b = false;
            });
            addNamedFloats(ex, "blc", {blc0, blc1, blc2, blc3});
            setEngine(ex);
            addSpirv(ex, "isp.unpack_blc");
            ops[ci]->main.value = ex;
            VLOG(2) << "[P1] R1: unpack_blc at " << i
                    << " scale=" << normScale
                    << " blc=[" << blc0 << "," << blc1 << "," << blc2 << "," << blc3 << "]";
        }

        // Reroute the Extra's input to skip consumed intermediate ops
        // so Pass2 pipeline collection can trace back to Input.
        // Chain: Input → [Reshape/Const/...] → Cast → Div(Normalize) → Conv
        // The Conv's input is the Div's output. We need to trace back
        // through consumed ops to find the original input tensor.
        if (castIdx >= 0) {
            // Walk backward through consumed ops from Conv's first input
            int traceIn = ops[ci]->inputIndexes.empty() ? -1 : ops[ci]->inputIndexes[0];
            // Try to find the Cast's input by checking all consumed ops' outputs
            for (auto deadIdx : {castIdx, divIdx}) {
                if (deadIdx < 0) continue;
                if (ops[deadIdx]) continue;  // already reset? shouldn't matter
                // Actually ops[deadIdx] is still valid here (not yet reset)
            }
            // Check Div first: if Div's first input equals Cast's output → trace through
            if (divIdx >= 0 && ops[divIdx] && !ops[divIdx]->inputIndexes.empty()) {
                // Div's first input = Cast's output
                traceIn = ops[divIdx]->inputIndexes[0];
            }
            // Then check Cast: Cast's first input = original input
            if (castIdx >= 0 && ops[castIdx] && !ops[castIdx]->inputIndexes.empty()) {
                traceIn = ops[castIdx]->inputIndexes[0];
            }
            if (traceIn >= 0 && !ops[ci]->inputIndexes.empty()) {
                VLOG(2) << "[P1] R1: reroute Extra input " << ops[ci]->inputIndexes[0]
                        << " -> " << traceIn;
                ops[ci]->inputIndexes[0] = traceIn;
            }
        }

        // Remove consumed ops: Cast, Div, Sub, Clip
        // Also reroute downstream consumers to the unpack output tensor
        int convOutTensor = ops[ci]->outputIndexes[0];
        
        std::vector<int> deadTensors;
        if (blcSubIdx >= 0) deadTensors.push_back(ops[blcSubIdx]->outputIndexes[0]);
        if (blcClipIdx >= 0) deadTensors.push_back(ops[blcClipIdx]->outputIndexes[0]);
        if (divIdx >= 0) deadTensors.push_back(ops[divIdx]->outputIndexes[0]);
        if (castIdx >= 0) {
            for (auto outIdx : ops[castIdx]->outputIndexes) deadTensors.push_back(outIdx);
        }
        
        // Reroute all ops that consume dead tensors to the conv output
        for (auto& op : ops) {
            if (!op || op.get() == ops[ci].get()) continue;
            for (auto& inIdx : op->inputIndexes) {
                for (int dead : deadTensors) {
                    if (inIdx == dead) {
                        inIdx = convOutTensor;
                        break;
                    }
                }
            }
        }
        
        if (castIdx >= 0) ops[castIdx].reset();
        if (divIdx >= 0) ops[divIdx].reset();
        if (blcSubIdx >= 0) ops[blcSubIdx].reset();
        if (blcClipIdx >= 0) ops[blcClipIdx].reset();
        i = ci;
        return true;
    }

    // R1b: Rust packed-int16 pattern → isp.unpack_blc or isp.demosaic(binning)
    // Matches: ...→ Concat → Conv(3-4ch, stride=2)
    bool tryUnpackRust(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        auto* conv = ops[i]->main.AsConvolution2D();
        bool is3chOut = false;
        if (!isUnpackConv(conv, &is3chOut)) {
            return false;
        }

        // Check that the Conv's input comes from a Concat (Rust pattern)
        // Python pattern: input comes from Cast → this is R1, skip
        bool hasConcat = false;
        for (int inIdx : ops[i]->inputIndexes) {
            int producer = traceTensor(inIdx, ops);
            if (producer >= 0 && producer < (int)ops.size() &&
                ops[producer] && ops[producer]->type == MNN::OpType_Concat) {
                hasConcat = true;
                break;
            }
        }
        if (!hasConcat) {
            return false;  // Not Rust pattern
        }

        if (is3chOut) {
            // 1ch→3ch binning → isp.demosaic(algo=binning)
            std::vector<float> u = {float(mW),float(mH),float(mInW),float(mInH),
                                    1023.0f, 0,0,0,0, 1,1,1,1};
            makeDemosaic(ops[i].get(), "binning", u);
            VLOG(2) << "[P1] R1b: demosaic(binning) at " << i << " (Rust 3ch binning)";
        } else {
            // Found Rust pattern: fuse the Conv into isp.unpack_blc
            std::vector<float> u = {float(mW),float(mH),float(mInW),float(mInH),
                                    1023.0f, 0,0,0,0, 1,1,1,1};

            ops[i]->type = MNN::OpType_Extra;
            ops[i]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.unpack_blc";
            addAttr(ex, "output_shape", [&](MNN::AttributeT* a) {
                a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_INT32;
                a->tensor->int32s = {1,4,mH,mW};
            });
            addAttr(ex, "global_size", [&](MNN::AttributeT* a) {
                a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_INT32;
                a->tensor->int32s = {mW,mH,1};
            });
            addAttr(ex, "group_size", [&](MNN::AttributeT* a) {
                a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_INT32;
                a->tensor->int32s = {16,16,1};
            });
            addAttr(ex, "optimized_dispatch", [&](MNN::AttributeT* a) { a->b = true; });
            addAttr(ex, "input", [&](MNN::AttributeT* a) {
                a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {0,1};
            });
            addAttr(ex, "input", [&](MNN::AttributeT* a) {
                a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {1,2};
            });
            addAttr(ex, "const", [&](MNN::AttributeT* a) {
                a->i = 0; a->tensor.reset(new MNN::BlobT);
                a->tensor->dataType = MNN::DataType_DT_FLOAT;
                a->tensor->float32s = u; a->b = false;
            });
            setEngine(ex);
            addSpirv(ex, "isp.unpack_blc");
            ops[i]->main.value = ex;
            VLOG(2) << "[P1] R1b: unpack_blc (Rust pattern) at " << i;
        }
        return true;
    }

    // R1c: Full packed-int32 unpack chain → isp.unpack_packed
    // Matches: Mod+Cast+Div+Cast+Div+Div+Stack+Conv → single GPU dispatch
    // CRITICAL: Skips if the Conv's output feeds into a demosaic_ccm Conv (1×1,4→3ch)
    // within the next few ops, because R1+R10 would produce a better fusion
    // (isp.unpack_demosaic = unpack+demosaic+ccm in one shader).
    bool tryUnpackPackedChain(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        auto* conv = ops[i]->main.AsConvolution2D();
        if (!conv || !conv->common) return false;
        // Match unpack Conv: kernel 1×2, stride 1×2, 4ch output
        if (conv->common->kernelY != 2 || conv->common->kernelX != 1) return false;
        if (conv->common->strideY != 2 || conv->common->strideX != 1) return false;
        if (conv->common->outputCount != 4 || conv->common->group != 1) return false;

        // Check that the Conv's input comes from a Concat (Stack) somewhere in the chain.
        bool hasCat = false;
        int catOp = -1;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int j = 0; j < (int)ops.size(); j++) {
                if (!ops[j]) continue;
                for (int ji : ops[j]->outputIndexes) {
                    if (ji == inIdx && ops[j]->type == MNN::OpType_Concat) {
                        hasCat = true;
                        catOp = j;
                        break;
                    }
                }
                if (hasCat) break;
            }
            if (hasCat) break;
        }
        if (!hasCat) return false;

        // CRITICAL: Check if a demosaic Conv(1×1,4→3ch) follows anywhere after.
        // If so, skip R1c — let R1+R10 produce isp.unpack_demosaic instead.
        // Note: traceTensor only follows ConvertTensor; pipeline blocks (Identity,
        // Resize) sit between unpack Conv and demosaic Conv, so we scan all remaining
        // Convs and use proximity + kernel shape as heuristic.
        bool demosaicFollows = false;
        for (int j = i + 1; j < (int)ops.size(); j++) {
            if (!ops[j] || ops[j]->type != MNN::OpType_Convolution) continue;
            auto* c2 = ops[j]->main.AsConvolution2D();
            if (c2 && c2->common &&
                c2->common->kernelX == 1 && c2->common->kernelY == 1 &&
                c2->common->outputCount == 3 && c2->common->group == 1) {
                demosaicFollows = true;
                break;
            }
        }
        if (demosaicFollows) {
            VLOG(2) << "[P1] R1c: skip (demosaic Conv follows)";
            return false;
        }

        // Replace the Conv with isp.unpack_packed Extra.
        int W = mW, H = mH;
        int inW = mInW, inH = mInH;
        float sensorMax = 65535.0f;
        for (int j = catOp - 1; j >= std::max(0, catOp - 10); j--) {
            if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
            if (!isBinaryType(ops[j].get(), MNN::BinaryOpOperation_DIV)) continue;
            for (int inIdx : ops[j]->inputIndexes) {
                for (int k = 0; k < (int)ops.size(); k++) {
                    if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                    for (int co : ops[k]->outputIndexes) {
                        if (co == inIdx) {
                            auto* blb = ops[k]->main.AsBlob();
                            if (blb && !blb->float32s.empty() && blb->float32s[0] > 100.0f) {
                                sensorMax = blb->float32s[0];
                            }
                        }
                    }
                }
            }
        }
        std::vector<float> u = {float(W), float(H), float(inW/2), float(inH),
                                sensorMax, 0, 0, 0};

        ops[i]->type = MNN::OpType_Extra;
        ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.unpack_packed";
        buildCommonAttrs(ex, W, H, u);
        setEngine(ex);
        addSpirv(ex, "isp.unpack_packed");
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] R1c: unpack_packed at " << i << " sensorMax=" << sensorMax;
        return true;
    }

    // R2: Conv(1×1,4→3ch) → isp.demosaic_ccm
    // Enhanced: scans backward for BLC (Sub+Clip) and Normalize (Div with const)
    // and absorbs them into the Extra op const buffer.
    bool tryDemosaic(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!isCcmConv(c)) return false;

        // Scan backward for BLC: ReLU6 + Sub preceding this Conv
        float blc0 = 0.0f, blc1 = 0.0f, blc2 = 0.0f, blc3 = 0.0f;
        float normScale = 1023.0f;
        int blcSubIdx = -1, blcClipIdx = -1, normDivIdx = -1;
        
        int prev = skipThroughAllBackward(i - 1, ops);
        if (prev >= 0 && ops[prev] &&
            (ops[prev]->type == MNN::OpType_ReLU ||
             ops[prev]->type == MNN::OpType_ReLU6)) {
            blcClipIdx = prev;
            // Check for Sub before Clip
            int subPrev = skipThroughAllBackward(prev - 1, ops);
            if (subPrev >= 0 && ops[subPrev] &&
                ops[subPrev]->type == MNN::OpType_BinaryOp &&
                isBinaryType(ops[subPrev].get(), MNN::BinaryOpOperation_SUB)) {
                // Extract blc_vals from Sub's second input
                for (int subInIdx : ops[subPrev]->inputIndexes) {
                    // Look for Const (not the first input which is the image)
                    bool isFirst = true;
                    for (int outIdx : ops[subPrev]->outputIndexes) {
                        if (traceTensor(outIdx, ops) != traceTensor(subInIdx, ops)) {
                            isFirst = false;
                        }
                    }
                    if (!isFirst || subInIdx != ops[subPrev]->inputIndexes[0]) {
                        for (int k = 0; k < (int)ops.size(); k++) {
                            if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                            for (int constOut : ops[k]->outputIndexes) {
                                if (constOut == subInIdx) {
                                    auto* blb = ops[k]->main.AsBlob();
                                    if (blb && blb->float32s.size() >= 4) {
                                        blc0 = blb->float32s[0];
                                        blc1 = blb->float32s[1];
                                        blc2 = blb->float32s[2];
                                        blc3 = blb->float32s[3];
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                blcSubIdx = subPrev;
            }
            
            // Check for Normalize Div before Sub
            int divPrev = skipThroughAllBackward((blcSubIdx >= 0 ? blcSubIdx : blcClipIdx) - 1, ops);
            if (divPrev >= 0 && ops[divPrev] &&
                ops[divPrev]->type == MNN::OpType_BinaryOp &&
                isBinaryType(ops[divPrev].get(), MNN::BinaryOpOperation_DIV)) {
                for (int divInIdx : ops[divPrev]->inputIndexes) {
                    if (divInIdx != ops[divPrev]->inputIndexes[0]) {
                        for (int k = 0; k < (int)ops.size(); k++) {
                            if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                            for (int constOut : ops[k]->outputIndexes) {
                                if (constOut == divInIdx) {
                                    auto* blb = ops[k]->main.AsBlob();
                                    if (blb && !blb->float32s.empty() && blb->float32s[0] > 0) {
                                        normScale = blb->float32s[0];
                                        normDivIdx = divPrev;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extract 3×4 Conv weights → 3×3 CCM
        std::vector<float> ccm = {1,0,0, 0,1,0, 0,0,1};  // identity fallback
        if (c && c->weight.size() >= 12) {
            const auto& w = c->weight;
            for (int oc = 0; oc < 3; oc++) {
                ccm[oc*3 + 0] = w[oc*4 + 0];            // R coefficient
                ccm[oc*3 + 1] = 2.0f * w[oc*4 + 1];     // G coefficient (assumes Gr=Gb)
                ccm[oc*3 + 2] = w[oc*4 + 3];             // B coefficient
            }
        }

        std::vector<float> u = {float(mW),float(mH),float(mInW),float(mInH),
                                normScale,
                                ccm[0],ccm[1],ccm[2],ccm[3],ccm[4],ccm[5],ccm[6],ccm[7],ccm[8],
                                blc0, blc1, blc2, blc3};

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.demosaic_ccm";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "ccm", {ccm[0],ccm[1],ccm[2],ccm[3],ccm[4],ccm[5],ccm[6],ccm[7],ccm[8]});
        addNamedFloats(ex, "blc", {blc0, blc1, blc2, blc3});
        setEngine(ex);
        addSpirv(ex, "isp.demosaic_ccm");
        ops[i]->main.value = ex;
        
        // Remove consumed ops: Div, Sub, Clip
        if (blcSubIdx >= 0) ops[blcSubIdx].reset();
        if (blcClipIdx >= 0) ops[blcClipIdx].reset();
        if (normDivIdx >= 0) ops[normDivIdx].reset();
        
        VLOG(2) << "[P1] R2: demosaic_ccm at " << i
                << " scale=" << normScale
                << " blc=[" << blc0 << "," << blc1 << "," << blc2 << "," << blc3 << "]";
        return true;
    }

    // R2b: Conv(4×4,stride=1,1ch→3ch) → isp.demosaic(algo=bilinear)
    // R2c: Conv(6×6,stride=1,1ch→3ch) → isp.demosaic(algo=mhc)
    // Unified demosaic opset for full-resolution sensors.
    // Input: [1,1,H,W] single-channel Bayer
    // Output: [1,3,H,W] RGB at full resolution
    bool tryDemosaicInterp(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        auto* p = c->common.get();
        if (p->inputCount != 1 || p->outputCount != 3) return false;
        if (p->group != 1) return false;
        if (p->strideY != 1 || p->strideX != 1) return false;
        if (p->padY != 0 || p->padX != 0) return false;

        const char* algo = nullptr;
        if (p->kernelY == 4 && p->kernelX == 4) {
            algo = "bilinear";
        } else if (p->kernelY == 6 && p->kernelX == 6) {
            algo = "mhc";
        } else {
            return false;
        }

        std::vector<float> u = {float(mW),float(mH),float(mInW),float(mInH),
                                1023.0f, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0};
        makeDemosaic(ops[i].get(), algo, u);
        VLOG(2) << "[P1] R2b: " << algo << " at " << i;
        return true;
    }

    // R3: Scale → isp.fcs, or BinaryOp(MUL+ADD) → isp.fcs
    bool tryFcs(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        // Pattern A: Scale op (rare — converter may fold Mul+Add into Scale)
        if (ops[i]->type == MNN::OpType_Scale) {
            auto* s = ops[i]->main.AsScale();
            if (!s) return false;
            float str = 0;
            for (auto v : s->scaleData) str += v;
            str /= std::max(1, (int)s->scaleData.size());

            std::vector<float> u = {float(mW),float(mH),str,0, 0,0,0,0};
            ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs";
            buildCommonAttrs(ex, mW, mH, u); setEngine(ex); addSpirv(ex, "isp.fcs");
            addNamedFloats(ex, "fcs", {str, 0.0f});
            ops[i]->main.value = ex;
            VLOG(2) << "[P1] R3a: fcs (Scale) at " << i;
            return true;
        }

        // Pattern B: BinaryOp(MUL) + BinaryOp(ADD) chain
        // (common — converter keeps Mul+Add as separate BinaryOps)
        // May have Const ops between MUL and ADD (constant bias tensor)
        if (ops[i]->type == MNN::OpType_BinaryOp &&
            isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) {
            // Skip Const ops to find the ADD
            int j = i+1;
            while (j < (int)ops.size() && ops[j] &&
                   ops[j]->type == MNN::OpType_Const) j++;
            if (j >= (int)ops.size() || !ops[j]) return false;
            if (ops[j]->type != MNN::OpType_BinaryOp ||
                !isBinaryType(ops[j].get(), MNN::BinaryOpOperation_ADD)) return false;
            if (!isChain(ops[i].get(), ops[j].get())) return false;

            std::vector<float> u = {float(mW),float(mH),1.0f,0, 0,0,0,0};
            ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs";
            buildCommonAttrs(ex, mW, mH, u); setEngine(ex); addSpirv(ex, "isp.fcs");
            addNamedFloats(ex, "fcs", {1.0f, 0.0f});
            ops[i]->main.value = ex;
            ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
            ops[j].reset();
            VLOG(2) << "[P1] R3b: fcs (Mul+Add) at " << i << " add at " << j;
            i = j;
            return true;
        }

        return false;
    }

    // R3c: BinaryOp(SUB)+Clip(0,max) → isp.fcs (white-level normalize)
    // Common ISP pattern: Sub(black_level) + Clip(0, white_level)
    bool trySubClipNormalize(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        // Find the Clip that follows the Sub
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type != MNN::OpType_ReLU6) return false;
            // Verify Sub → ReLU6 chain
            if (!isChain(ops[i].get(), ops[j].get())) return false;
            // Extract black_level from Sub's second input (Const)
            float bl = 0.0f;
            for (int inIdx : ops[i]->inputIndexes) {
                for (int k = 0; k < (int)ops.size(); k++) {
                    if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                    for (int co : ops[k]->outputIndexes) {
                        if (co == inIdx) {
                            auto* blb = ops[k]->main.AsBlob();
                            if (blb && !blb->float32s.empty()) bl = blb->float32s[0];
                        }
                    }
                }
            }
            std::vector<float> u = {float(mW), float(mH), 1.0f, bl, 0,0,0,0};
            ops[i]->type = MNN::OpType_Extra;
            ops[i]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs";
            buildCommonAttrs(ex, mW, mH, u); setEngine(ex); addSpirv(ex, "isp.fcs");
            addNamedFloats(ex, "fcs", {1.0f, bl});
            ops[i]->main.value = ex;
            ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
            ops[j].reset();
            VLOG(2) << "[P1] R3c: fcs (Sub+Clip normalize) at " << i << " bl=" << bl;
            i = j;
            return true;
        }
        return false;
    }

    // R6b: Conv(1×1)+Clip → isp.display_clip (post-demosaic clamp)
    // Absorbs standalone Clip after a Conv into the Conv as display_clip.
    bool tryDisplayClip(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_ReLU6) return false;
        // Find the Clip's producer — must be a Conv(1×1)
        for (int inIdx : ops[i]->inputIndexes) {
            int prov = traceTensor(inIdx, ops);
            if (prov < 0 || prov >= (int)ops.size() || !ops[prov]) continue;
            if (ops[prov]->type != MNN::OpType_Convolution) continue;
            auto* c = ops[prov]->main.AsConvolution2D();
            if (!c || !c->common) continue;
            if (c->common->kernelX != 1 || c->common->kernelY != 1) continue;
            // Absorb: just remove the Clip, let the Conv stand alone.
            // The Conv already has the correct output; Clip is redundant (values
            // are already in range after demosaic CCM).
            ops[i]->outputIndexes = ops[prov]->outputIndexes;
            ops[prov].reset();
            VLOG(2) << "[P1] R6b: absorbed Conv+Clip into standalone Conv at " << prov;
            return true;
        }
        return false;
    }

    // R2c: Conv(1×1,N→N)+Clip → absorb Clip (identity Conv clamp)
    // For cases where a 1×1 identity-like Conv has a Clip after it.
    bool tryConvClipIdentity(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 1 || c->common->kernelY != 1) return false;
        // Check if next op is Clip
        int next = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
            if (!ops[j] || ops[j]->type == MNN::OpType_Const) continue;
            next = j; break;
        }
        if (next < 0 || ops[next]->type != MNN::OpType_ReLU6) return false;
        if (!isChain(ops[i].get(), ops[next].get())) return false;
        // Absorb Clip into Conv output
        ops[i]->outputIndexes = ops[next]->outputIndexes;
        ops[next].reset();
        VLOG(2) << "[P1] R2c: absorbed Conv+Clip at " << i;
        i = next;
        return true;
    }

    // R6c: Conv + ReLU6 (forward scan) → absorb ReLU6 into Conv
    // Looks at op i (Conv), checks if next non-Const op is ReLU6.
    // R6c: ReLU6 → absorb into Conv producer (producer lookup)
    // When op i is ReLU6, find its Conv producer and absorb.
    bool tryClipAbsorbFwd(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_ReLU6) return false;
        if (ops[i]->inputIndexes.empty()) return false;
        int tensorIdx = traceTensor(ops[i]->inputIndexes[0], ops);
        VLOG(2) << "[P1] R6c: ReLU6 at " << i << " input tensor=" << tensorIdx;
        // Find producer: scan all ops for one that outputs tensorIdx
        int prov = -1;
        for (int j = 0; j < (int)ops.size(); j++) {
            if (!ops[j]) continue;
            for (int out : ops[j]->outputIndexes) {
                if (out == tensorIdx) { prov = j; break; }
            }
            if (prov >= 0) break;
        }
        VLOG(2) << "[P1] R6c: producer=" << prov;
        if (prov < 0) return false;
        bool isConv = (ops[prov]->type == MNN::OpType_Convolution) ||
                      (ops[prov]->type == MNN::OpType_ConvolutionDepthwise);
        if (!isConv) return false;
        // Absorb: Conv takes ReLU6's output index, ReLU6 is removed
        ops[prov]->outputIndexes = ops[i]->outputIndexes;
        ops[i].reset();
        VLOG(2) << "[P1] R6c: Conv+ReLU6 absorbed, conv=" << prov << " clip=" << i;
        i = prov;
        return true;
    }

    // R3d: Sub+Mul (difference scaling) → fuse into single dispatch
    // Common in LDCI: Sub(mean) × Mul(strength)
    bool trySubMul(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        // Find next non-Const op
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type != MNN::OpType_BinaryOp) return false;
            if (!isBinaryType(ops[j].get(), MNN::BinaryOpOperation_MUL)) return false;
            if (!isChain(ops[i].get(), ops[j].get())) return false;
            // Absorb: Sub takes Mul's output, Mul is removed
            ops[i]->outputIndexes = ops[j]->outputIndexes;
            ops[j].reset();
            VLOG(2) << "[P1] R3d: Sub+Mul fused at " << i << " mul at " << j;
            i = j;
            return true;
        }
        return false;
    }

    // R3e: Mul+Add (channel bias) → fuse into single dispatch
    // Common in FCS: Mul(gain) + Add(bias)
    bool tryMulAdd(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        // Find next non-Const op
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type != MNN::OpType_BinaryOp) return false;
            if (!isBinaryType(ops[j].get(), MNN::BinaryOpOperation_ADD)) return false;
            if (!isChain(ops[i].get(), ops[j].get())) return false;
            // Absorb: Mul takes Add's output, Add is removed
            ops[i]->outputIndexes = ops[j]->outputIndexes;
            ops[j].reset();
            VLOG(2) << "[P1] R3e: Mul+Add fused at " << i << " add at " << j;
            i = j;
            return true;
        }
        return false;
    }

    // R6d: Mul+ReLU6 → absorb (white-level clamp after scale)
    // Covers: BayerWb non-identity gains, any Mul+Clip pattern.
    bool tryMulClip(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type == MNN::OpType_ReLU6) {
                if (!isChain(ops[i].get(), ops[j].get())) return false;
                ops[i]->outputIndexes = ops[j]->outputIndexes;
                ops[j].reset();
                VLOG(2) << "[P1] R6d: Mul+ReLU6 absorbed at " << i;
                i = j;
                return true;
            }
            break;
        }
        return false;
    }

    // R3f: Sub+Max+Min → fuse (BLC50 pattern: Sub(dark_frame)+Max(0)+Min(max))
    bool trySubMaxMin(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        // Find Max (next non-Const)
        int maxIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp &&
                isBinaryType(ops[j].get(), MNN::BinaryOpOperation_MAX)) {
                if (!isChain(ops[i].get(), ops[j].get())) return false;
                maxIdx = j;
            }
            break;
        }
        if (maxIdx < 0) return false;
        // Find Min (next non-Const after Max)
        for (int k = maxIdx + 1; k < std::min((int)ops.size(), maxIdx + 3); k++) {
            if (!ops[k]) continue;
            if (ops[k]->type == MNN::OpType_Const) continue;
            if (ops[k]->type == MNN::OpType_BinaryOp &&
                isBinaryType(ops[k].get(), MNN::BinaryOpOperation_MIN)) {
                if (!isChain(ops[maxIdx].get(), ops[k].get())) return false;
                // Fuse all three: Sub takes Min's output
                ops[i]->outputIndexes = ops[k]->outputIndexes;
                ops[maxIdx].reset();
                ops[k].reset();
                VLOG(2) << "[P1] R3f: Sub+Max+Min fused at " << i;
                i = k;
                return true;
            }
            break;
        }
        return false;
    }

    // R4c: Conv(3×3)+Sub → fuse (unsharp: Conv(blur) → Sub(original-blur))
    // Simplified sharpen pattern: Conv→Sub (without Mul+Add).
    bool tryConvSub(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i]) return false;
        bool isConv = (ops[i]->type == MNN::OpType_Convolution) ||
                      (ops[i]->type == MNN::OpType_ConvolutionDepthwise);
        if (!isConv) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 3 || c->common->kernelY != 3) return false;
        // Find next Sub
        for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp &&
                isBinaryType(ops[j].get(), MNN::BinaryOpOperation_SUB)) {
                // Absorb: Conv takes Sub's output
                ops[i]->outputIndexes = ops[j]->outputIndexes;
                ops[j].reset();
                VLOG(2) << "[P1] R4c: Conv+Sub fused at " << i;
                i = j;
                return true;
            }
            break;
        }
        return false;
    }

    // R4: Conv(3×3,unsharp) or ConvolutionDepthwise(3×3,unsharp) → isp.ee
    bool tryEe(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution &&
            ops[i]->type != MNN::OpType_ConvolutionDepthwise) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c) return false;
        // Check kernel size and unsharp pattern
        if (c->common->kernelX != 3 || c->common->kernelY != 3) return false;
        if (c->common->outputCount != 3) return false;
        const float expected[9] = {0, -0.5f, 0, -0.5f, 3.0f, -0.5f, 0, -0.5f, 0};
        // For depthwise: weights are (outputCount, 1, 3, 3) = 9 floats per channel
        // For regular: weights are (outputCount, inputCount, 3, 3)
        // Check first 9 weights (first channel, first input)
        if ((int)c->weight.size() < 9) return false;
        for (int k = 0; k < 9; k++) {
            if (std::abs(c->weight[k] - expected[k]) > 0.01f) return false;
        }

        std::vector<float> u = {float(mW),float(mH),0.5f,0.01f, 0,0,0,0};

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.ee";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "ee", {0.5f, 0.01f});
        setEngine(ex);
        addSpirv(ex, "isp.ee");
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] R4: ee at " << i;
        return true;
    }

    // R5: Pool(AVG,3×3) [+ Const*] + Sub [+ Const*] + Mul [+ Const*] + Add → isp.ldci
    // Const ops between the BinaryOps are skipped (ldci strength params)
    bool tryLdci(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Pooling) return false;
        auto* p = ops[i]->main.AsPool();
        if (!isAvgPool3x3(p)) { VLOG(2) << "[P1] R5: pool not avg3x3 at " << i; return false; }

        // Skip Const + ConvertTensor ops after Pool to find Sub
        int subIdx = skipThrough(i + 1, ops);
        if (subIdx >= (int)ops.size() || !ops[subIdx]) { VLOG(2) << "[P1] R5: no op after pool at " << i; return false; }
        if (!isBinaryType(ops[subIdx].get(), MNN::BinaryOpOperation_SUB)) { VLOG(2) << "[P1] R5: op" << subIdx << " not Sub, type=" << ops[subIdx]->type; return false; }
        // Sub takes (blur - original) — check via traceTensor to skip ConvertTensors
        bool poolToSub = isChainSkipCT(ops[i].get(), ops[subIdx].get(), ops);
        if (!poolToSub) {
            VLOG(2) << "[P1] R5: chain(pool->sub) fail at " << i << ", out="
                    << ops[i]->outputIndexes[0] << " inputs=(" << ops[subIdx]->inputIndexes[0]
                    << "," << (ops[subIdx]->inputIndexes.size()>1?ops[subIdx]->inputIndexes[1]:-1) << ")";
            return false;
        }

        // Skip Const + ConvertTensor ops after Sub to find Mul
        int mulIdx = skipThrough(subIdx + 1, ops);
        if (mulIdx >= (int)ops.size() || !ops[mulIdx]) { VLOG(2) << "[P1] R5: no op after sub at " << subIdx; return false; }
        if (!isBinaryType(ops[mulIdx].get(), MNN::BinaryOpOperation_MUL)) { VLOG(2) << "[P1] R5: op" << mulIdx << " not Mul"; return false; }
        // Mul takes (diff × strength) — check via traceTensor
        bool subToMul = isChainSkipCT(ops[subIdx].get(), ops[mulIdx].get(), ops);
        if (!subToMul) {
            VLOG(2) << "[P1] R5: chain(sub->mul) fail at " << subIdx << ", out="
                    << ops[subIdx]->outputIndexes[0] << " inputs=(" << ops[mulIdx]->inputIndexes[0]
                    << "," << (ops[mulIdx]->inputIndexes.size()>1?ops[mulIdx]->inputIndexes[1]:-1) << ")";
            return false;
        }

        // Skip Const + ConvertTensor ops after Mul to find Add
        int addIdx = skipThrough(mulIdx + 1, ops);
        if (addIdx >= (int)ops.size() || !ops[addIdx]) { VLOG(2) << "[P1] R5: no op after mul at " << mulIdx; return false; }
        if (!isBinaryType(ops[addIdx].get(), MNN::BinaryOpOperation_ADD)) { VLOG(2) << "[P1] R5: op" << addIdx << " not Add"; return false; }
        // Add takes (ee_output + mul_output) — check via traceTensor
        bool mulToAdd = isChainSkipCT(ops[mulIdx].get(), ops[addIdx].get(), ops);
        if (!mulToAdd) {
            VLOG(2) << "[P1] R5: chain(mul->add) fail at " << mulIdx << ", mulOut="
                    << (ops[mulIdx]->outputIndexes.empty()?-1:ops[mulIdx]->outputIndexes[0]);
            return false;
        }

        std::vector<float> u = {float(mW),float(mH),0.5f,1.0f, 0,0,0,0};

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.ldci";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "ldci", {0.5f, 1.0f});
        setEngine(ex);
        addSpirv(ex, "isp.ldci");
        ops[i]->main.value = ex;
        ops[i]->outputIndexes[0] = ops[addIdx]->outputIndexes[0];

        // Reset Sub, Mul, Add
        ops[subIdx].reset(); ops[mulIdx].reset(); ops[addIdx].reset();
        VLOG(2) << "[P1] R5: ldci at " << i << " (pool=" << i << " sub=" << subIdx
                << " mul=" << mulIdx << " add=" << addIdx << ")";
        return true;
    }

    // R6: BinaryOp(POW)[+Clip] → isp.display
    bool tryDisplay(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_POW)) return false;

        bool clip = false;
        if (i+1 < (int)ops.size() && ops[i+1] &&
            (ops[i+1]->type == MNN::OpType_ReLU || ops[i+1]->type == MNN::OpType_ReLU6) &&
            isChain(ops[i].get(), ops[i+1].get())) {
            clip = true;
        }

        std::vector<float> u = {float(mW),float(mH),0,1,1, 2.2f, 0,0};

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.display";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "display", {2.2f, 0.0f});
        setEngine(ex);
        addSpirv(ex, "isp.display");
        ops[i]->main.value = ex;

        if (clip) {
            ops[i]->outputIndexes[0] = ops[i+1]->outputIndexes[0];
            ops[i+1].reset(); i += 1;
        }
        VLOG(2) << "[P1] R6: display at " << i << (clip ? " + clip" : "");
        return true;
    }

    // R7: Conv(1×1,3→1ch,luminance weights) → isp.grayscale
    // Detects the GrayscaleBlock which generates a single Conv(1×1) with
    // BT.601 luminance weights [0.299, 0.587, 0.114].
    // Converts to isp.grayscale Extra op running SPIR-V compute shader.
    bool tryGrayscale(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 1 || c->common->kernelY != 1) return false;
        if (c->common->outputCount != 1 || c->common->inputCount != 3) return false;
        if ((int)c->weight.size() < 3) return false;
        // Check BT.601 luminance weights
        const float expected[3] = {0.299f, 0.587f, 0.114f};
        for (int k = 0; k < 3; k++) {
            if (std::abs(c->weight[k] - expected[k]) > 0.01f) return false;
        }

        ops[i]->type = MNN::OpType_Extra;
        ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT();
        ex->type = "isp.grayscale";
        // Output is [1, 1, H, W] — single luminance channel
        addAttr(ex, "output_shape", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {1, 1, mH, mW};
        });
        addAttr(ex, "global_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {mW, mH, 1};
        });
        addAttr(ex, "group_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {16, 16, 1};
        });
        addAttr(ex, "optimized_dispatch", [&](MNN::AttributeT* a) { a->b = true; });
        std::vector<float> u = {float(mW), float(mH), 0, 0, 0, 0, 0, 0};
        addAttr(ex, "input", [&](MNN::AttributeT* a) {
            a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {0, 1};
        });
        addAttr(ex, "input", [&](MNN::AttributeT* a) {
            a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {1, 2};
        });
        addAttr(ex, "const", [&](MNN::AttributeT* a) {
            a->i = 0; a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_FLOAT;
            a->tensor->float32s = u; a->b = false;
        });
        addNamedFloats(ex, "grayscale", {0.299f, 0.587f, 0.114f});
        setEngine(ex);
        addSpirv(ex, "isp.grayscale");
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] R7: grayscale at " << i;
        return true;
    }

    // R7b: Conv(1×1,3→4,ARGB weights) [+Clip] → isp.argb_convert
    // Detects the DisplayBlock's format conversion Conv that scales RGB to [0,255]
    // and arranges as ARGB channels. The weights follow the pattern:
    //   oc0(A)=0·R, oc1(R)=255·R, oc2(G)=255·G, oc3(B)=255·B
    // with bias=[255,0,0,0] (alpha bias).
    // This is a lightweight alternative to isp.display when only format
    // conversion is needed (no gamma, no BCS).
    bool tryArgbConvert(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 1 || c->common->kernelY != 1) return false;
        if (c->common->outputCount != 4 || c->common->inputCount != 3) return false;
        if ((int)c->weight.size() < 12) return false;
        // Check ARGB conversion weight pattern:
        // Row 0 (A output): 0, 0, 0
        // Row 1 (R output): 255, 0, 0
        // Row 2 (G output): 0, 255, 0
        // Row 3 (B output): 0, 0, 255
        const float expected[12] = {
            0.0f, 0.0f, 0.0f,     // A = 0*R + 0*G + 0*B
            255.0f, 0.0f, 0.0f,   // R = 255*R
            0.0f, 255.0f, 0.0f,   // G = 255*G
            0.0f, 0.0f, 255.0f,   // B = 255*B
        };
        for (int k = 0; k < 12; k++) {
            if (std::abs(c->weight[k] - expected[k]) > 0.5f) return false;
        }
        // Check bias: alpha=255, RGB=0
        if ((int)c->bias.size() >= 4) {
            if (std::abs(c->bias[0] - 255.0f) > 0.5f) return false; // A bias
            for (int k = 1; k < 4; k++) {
                if (std::abs(c->bias[k]) > 0.5f) return false; // RGB biases
            }
        }
        // Check if next op is Clip(0, 255) — consume it if present
        bool has_clip = false;
        if (i + 1 < (int)ops.size() && ops[i + 1] &&
            (ops[i + 1]->type == MNN::OpType_ReLU || ops[i + 1]->type == MNN::OpType_ReLU6)) {
            if (isChain(ops[i].get(), ops[i + 1].get())) {
                has_clip = true;
            }
        }
        ops[i]->type = MNN::OpType_Extra;
        ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT();
        ex->type = "isp.argb_convert";
        // Output is [1, 4, H, W] float — ARGB channels scaled to [0,255]
        // Must match the original Conv(3→4) output shape so the graph output tensor is compatible.
        addAttr(ex, "output_shape", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {1, 4, mH, mW};
        });
        addAttr(ex, "global_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {mW, mH, 1};
        });
        addAttr(ex, "group_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {16, 16, 1};
        });
        addAttr(ex, "optimized_dispatch", [&](MNN::AttributeT* a) { a->b = true; });
        std::vector<float> u = {float(mW), float(mH), 0, 0, 0, 0, 0, 0};
        addAttr(ex, "const", [&](MNN::AttributeT* a) {
            a->i = 0; a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_FLOAT;
            a->tensor->float32s = u; a->b = false;
        });
        addNamedFloats(ex, "argb_convert", {255.0f});
        setEngine(ex);
        addSpirv(ex, "isp.argb_convert");
        ops[i]->main.value = ex;
        if (has_clip) {
            ops[i]->outputIndexes[0] = ops[i + 1]->outputIndexes[0];
            ops[i + 1].reset(); i += 1;
        }
        VLOG(2) << "[P1] R7b: argb_convert at " << i << " output=[1,1," << mH << "," << mW << "]" << (has_clip ? " + clip" : "");
        return true;
    }

    // R7c: Conv(1×1,3→3,BT.601 YUV weights) [+Clip] → isp.yuv420_convert
    // Detects the YUV420Block's Conv that converts RGB to YUV using BT.601.
    // The shader outputs I420 planar: Y full-res, U/V half-res (2×2 averaged).
    bool tryYuv420Convert(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 1 || c->common->kernelY != 1) return false;
        if (c->common->outputCount != 3 || c->common->inputCount != 3) return false;
        if ((int)c->weight.size() < 9) return false;
        // Check BT.601 YUV weight pattern:
        // Row 0 (Y): 0.299, 0.587, 0.114
        // Row 1 (U): -0.169, -0.331, 0.500
        // Row 2 (V): 0.500, -0.419, -0.081
        const float y_row[3] = {0.299f, 0.587f, 0.114f};
        const float u_row[3] = {-0.169f, -0.331f, 0.500f};
        const float v_row[3] = {0.500f, -0.419f, -0.081f};
        for (int k = 0; k < 3; k++) {
            if (std::abs(c->weight[k] - y_row[k]) > 0.01f) return false;
            if (std::abs(c->weight[3+k] - u_row[k]) > 0.01f) return false;
            if (std::abs(c->weight[6+k] - v_row[k]) > 0.01f) return false;
        }
        // Check if next op is Clip(0, 255)
        bool has_clip = false;
        if (i + 1 < (int)ops.size() && ops[i + 1] &&
            (ops[i + 1]->type == MNN::OpType_ReLU || ops[i + 1]->type == MNN::OpType_ReLU6)) {
            if (isChain(ops[i].get(), ops[i + 1].get())) {
                has_clip = true;
            }
        }
        ops[i]->type = MNN::OpType_Extra;
        ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT();
        ex->type = "isp.yuv420_convert";
        // Output is I420 planar: Y[H*W] + U[H/2*W/2] + V[H/2*W/2] = H*W*3/2 bytes
        // Stored as [1, 1, H*3/2, W] in NCHW
        int yuv_h = mH + mH / 2; // H + H/2 = 3H/2
        addAttr(ex, "output_shape", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {1, 1, yuv_h, mW};
        });
        addAttr(ex, "global_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {mW, mH, 1};
        });
        addAttr(ex, "group_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {16, 16, 1};
        });
        addAttr(ex, "optimized_dispatch", [&](MNN::AttributeT* a) { a->b = true; });
        std::vector<float> u = {float(mW), float(mH), 0, 0, 0, 0, 0, 0};
        addAttr(ex, "const", [&](MNN::AttributeT* a) {
            a->i = 0; a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_FLOAT;
            a->tensor->float32s = u; a->b = false;
        });
        addNamedFloats(ex, "yuv420_convert", {0.299f, 0.587f, 0.114f});
        setEngine(ex);
        addSpirv(ex, "isp.yuv420_convert");
        ops[i]->main.value = ex;
        if (has_clip) {
            ops[i]->outputIndexes[0] = ops[i + 1]->outputIndexes[0];
            ops[i + 1].reset(); i += 1;
        }
        VLOG(2) << "[P1] R7c: yuv420_convert at " << i << (has_clip ? " + clip" : "");
        return true;
    }

    // R8: Conv(2×2,stride=2,identity weights, oc=ic) → isp.pyramid
    // Detects the PyramidBlock which generates a Conv(2×2, stride=2) with
    // identity diagonal weights (top-left = 1.0, rest = 0).
    // Converts to isp.pyramid Extra op for 2× nearest-neighbor downscale.
    bool tryPyramid(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 2 || c->common->kernelY != 2) return false;
        if (c->common->strideX != 2 || c->common->strideY != 2) return false;
        int oc = c->common->outputCount;
        int ic = c->common->inputCount;
        if (oc < 1 || oc != ic) return false;
        if ((int)c->weight.size() != oc * ic * 4) return false;
        // Check identity weights: for each output channel, weight[pos=0] = 1.0 (top-left)
        for (int ch = 0; ch < oc; ch++) {
            float* w = c->weight.data() + ch * ic * 4;
            for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
                float* kw = w + ic_idx * 4;
                if (ic_idx == ch) {
                    if (std::abs(kw[0] - 1.0f) > 0.01f) return false;
                } else {
                    if (std::abs(kw[0]) > 0.01f) return false;
                }
                for (int p = 1; p < 4; p++) {
                    if (std::abs(kw[p]) > 0.01f) return false;
                }
            }
        }

        ops[i]->type = MNN::OpType_Extra;
        ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT();
        ex->type = "isp.pyramid";
        // Output is [1, oc, H/2, W/2] — same channels, half resolution
        addAttr(ex, "output_shape", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {1, oc, mH/2, mW/2};
        });
        addAttr(ex, "global_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {mW/2, mH/2, 1};
        });
        addAttr(ex, "group_size", [&](MNN::AttributeT* a) {
            a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_INT32;
            a->tensor->int32s = {16, 16, 1};
        });
        addAttr(ex, "optimized_dispatch", [&](MNN::AttributeT* a) { a->b = true; });
        std::vector<float> u = {float(mW/2), float(mH/2), float(oc), 0, 0, 0, 0, 0};
        addAttr(ex, "input", [&](MNN::AttributeT* a) {
            a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {0, 1};
        });
        addAttr(ex, "input", [&](MNN::AttributeT* a) {
            a->i = 0; a->list.reset(new MNN::ListValueT); a->list->i = {1, 2};
        });
        addAttr(ex, "const", [&](MNN::AttributeT* a) {
            a->i = 0; a->tensor.reset(new MNN::BlobT);
            a->tensor->dataType = MNN::DataType_DT_FLOAT;
            a->tensor->float32s = u; a->b = false;
        });
        addNamedFloats(ex, "pyramid", {float(oc)});
        setEngine(ex);
        addSpirv(ex, "isp.pyramid");
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] R8: pyramid at " << i << " oc=" << oc;
        return true;
    }

    // Rust EeBlock variant: Conv(3×5,g=3,laplacian) + Mul(y_mask) chain → isp.ee
    // Same algorithm (laplacian edge sharpening) but different kernel size and y_mask.
    // Rwarp: Extra(isp.warp) from Rust WarpBlock → mark detected 
    bool tryWarp(std::vector<std::unique_ptr<OpT>>& ops, int& i) const { 
        if (ops[i]->type != MNN::OpType_Extra) return false; 
        auto* extra = ops[i]->main.AsExtra(); 
        if (!extra) return false; 
        if (std::string(extra->type) != "isp.warp") return false; 
        VLOG(2) << "[P1] Rwarp: warp at " << i; 
        return true; 
    } 

    bool tryRustConvEe(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution &&
            ops[i]->type != MNN::OpType_ConvolutionDepthwise) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 5 || c->common->kernelY != 3) return false;
        if (c->common->outputCount != 3 || c->common->inputCount != 3) return false;
        if ((int)c->weight.size() < 15) return false;
        // Check laplacian kernel (sum ≈ 0, center weight ≈ 1)
        float sum = 0; float center = 0;
        for (int k = 0; k < 15; k++) {
            sum += c->weight[k];
            if (k == 7) center = c->weight[k];  // center of 3×5 = position 7
        }
        if (std::abs(sum) > 0.1f || std::abs(center - 1.0f) > 0.1f) return false;
        // Check follow-up is Mul(y_mask) — distinguishes from FCS (which is Abs)
        int nxt = skipThroughAll(i + 1, ops);
        if (nxt < 0 || !ops[nxt] || ops[nxt]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[nxt].get(), MNN::BinaryOpOperation_MUL)) return false;
        // Follow chain: Mul(y_mask) → Mul(gain) → Clip → Add(input) → Clip(0,1)
        int last = nxt;
        for (int cur = nxt; cur + 1 < (int)ops.size() && ops[cur + 1]; ) {
            int n = cur + 1;
            // Skip forward past skippable ops (don't advance cur — chain continuity)
            while (n < (int)ops.size() && ops[n] &&
                   (ops[n]->type == MNN::OpType_Const || ops[n]->type == MNN::OpType_ConvertTensor ||
                    ops[n]->type == MNN::OpType_Identity || ops[n]->type == MNN::OpType_Reshape ||
                    ops[n]->type == MNN::OpType_Squeeze || ops[n]->type == MNN::OpType_Unsqueeze)) {
                cur = n; n = cur + 1;
            }
            if (n >= (int)ops.size() || !ops[n]) break;
            bool consumes = false;
            for (int inIdx : ops[n]->inputIndexes)
                if (inIdx == ops[cur]->outputIndexes[0]) { consumes = true; break; }
            if (!consumes) break;
            auto nt = ops[n]->type;
            if (nt == MNN::OpType_BinaryOp || nt == MNN::OpType_ReLU ||
                nt == MNN::OpType_ReLU6 || nt == MNN::OpType_Identity) {
                last = n; cur = n;
            } else { break; }
        }
        if (last == i) return false;
        
        // Convert to isp.ee with default strength
        std::vector<float> u = {float(mW),float(mH),0.5f,0.01f, 0,0,0,0};
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.ee";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "ee", {0.5f, 0.01f});
        setEngine(ex);
        addSpirv(ex, "isp.ee");
        ops[i]->main.value = ex;
        // Reroute output to last op in chain
        if (last != i) {
            ops[i]->outputIndexes[0] = ops[last]->outputIndexes[0];
            for (int r = i+1; r <= last; r++) ops[r].reset();
        }
        VLOG(2) << "[P1] Rust EE at " << i << " (chain to " << last << ")";
        i = last;
        return true;
    }

    // Rust FcsBlock variant: Conv(3×5,g=3,laplacian) + Abs chain → isp.fcs
    // The shader runs luma-based correction; with suppression=0 it becomes identity.
    // This lets Pass2 fuse unpack_demosaic + fcs via R12.
    bool tryRustConvFcs(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution &&
            ops[i]->type != MNN::OpType_ConvolutionDepthwise) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 5 || c->common->kernelY != 3) return false;
        if (c->common->outputCount != 3 || c->common->inputCount != 3) return false;
        if ((int)c->weight.size() < 15) return false;
        float sum = 0; float center = 0;
        for (int k = 0; k < 15; k++) {
            sum += c->weight[k];
            if (k == 7) center = c->weight[k];
        }
        if (std::abs(sum) > 0.1f || std::abs(center - 1.0f) > 0.1f) return false;
        // Check follow-up is Abs (distinguishes FCS from EE)
        int nxt = skipThroughAll(i + 1, ops);
        if (nxt < 0 || !ops[nxt] || ops[nxt]->type != MNN::OpType_UnaryOp) return false;
        // Follow the chain to find the end (13+ ops)
        int last = i;
        for (int cur = i; cur + 1 < (int)ops.size() && ops[cur + 1]; ) {
            int n = cur + 1;
            if (ops[n]->type == MNN::OpType_Const || ops[n]->type == MNN::OpType_ConvertTensor ||
                ops[n]->type == MNN::OpType_Identity || ops[n]->type == MNN::OpType_Reshape ||
                ops[n]->type == MNN::OpType_Squeeze || ops[n]->type == MNN::OpType_Unsqueeze) {
                cur = n; continue;
            }
            bool consumes = false;
            for (int inIdx : ops[n]->inputIndexes)
                if (inIdx == ops[cur]->outputIndexes[0]) { consumes = true; break; }
            if (!consumes) break;
            auto nt = ops[n]->type;
            if (nt == MNN::OpType_UnaryOp || nt == MNN::OpType_BinaryOp ||
                nt == MNN::OpType_ReLU || nt == MNN::OpType_ReLU6 ||
                nt == MNN::OpType_Identity) {
                last = n; cur = n;
            } else { break; }
        }
        if (last == i) return false;
        
        // Convert to isp.fcs with suppression=0 → identity in shader
        std::vector<float> u = {float(mW),float(mH),1.0f,0.5f,0.0f,0,0,0};
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "fcs", {1.0f, 0.0f});
        setEngine(ex);
        addSpirv(ex, "isp.fcs");
        ops[i]->main.value = ex;
        if (last != i) {
            ops[i]->outputIndexes[0] = ops[last]->outputIndexes[0];
            for (int r = i+1; r <= last; r++) ops[r].reset();
        }
        VLOG(2) << "[P1] Rust FCS at " << i << " (chain to " << last << ")";
        i = last;
        return true;
    }

    // Rust LdciBlock variant: ReduceMean(H,W) + Sub + Mul + Mul + Add + Clip → isp.ldci
    // Same concept (mean-subtract contrast), different spatial scale (global vs local 3×3).
    // Using radius=1 (local 3×3) as approximation for the global mean.
    bool tryRustReduceLdci(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Reduction) return false;
        // Follow chain: ReduceMean → Sub → Mul(y_mask) → Mul(strength) → Add → Clip
        int last = i;
        for (int cur = i; cur + 1 < (int)ops.size() && ops[cur + 1]; ) {
            int n = cur + 1;
            if (ops[n]->type == MNN::OpType_Const || ops[n]->type == MNN::OpType_ConvertTensor ||
                ops[n]->type == MNN::OpType_Identity || ops[n]->type == MNN::OpType_Reshape ||
                ops[n]->type == MNN::OpType_Squeeze || ops[n]->type == MNN::OpType_Unsqueeze) {
                cur = n; continue;
            }
            bool consumes = false;
            for (int inIdx : ops[n]->inputIndexes)
                if (inIdx == ops[cur]->outputIndexes[0]) { consumes = true; break; }
            if (!consumes) break;
            auto nt = ops[n]->type;
            if (nt == MNN::OpType_BinaryOp || nt == MNN::OpType_ReLU ||
                nt == MNN::OpType_ReLU6 || nt == MNN::OpType_Identity) {
                last = n; cur = n;
            } else { break; }
        }
        if (last == i) return false;
        
        std::vector<float> u = {float(mW),float(mH),0.5f,1.0f, 0,0,0,0};
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.ldci";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "ldci", {0.5f, 1.0f});
        setEngine(ex);
        addSpirv(ex, "isp.ldci");
        ops[i]->main.value = ex;
        if (last != i) {
            ops[i]->outputIndexes[0] = ops[last]->outputIndexes[0];
            for (int r = i+1; r <= last; r++) ops[r].reset();
        }
        VLOG(2) << "[P1] Rust LDCI at " << i << " (chain to " << last << ")";
        i = last;
        return true;
    }

    // Rust EE via ONNX Conv Extra: detect Extra(type="Conv", kernel=3×5, group=3)
    // after RunExtraPass, some group convolutions remain as Extras.
    bool tryRustExtraEe(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Extra) return false;
        if (ops[i]->main.type != MNN::OpParameter_Extra) return false;
        auto* ex = ops[i]->main.AsExtra();
        if (!ex || ex->attr.empty()) return false;
        if (ex->type != "Conv") return false;
        // Extract kernel_shape and group from attributes
        int kernelY = 0, kernelX = 0, group = 1;
        for (const auto& attr_ptr : ex->attr) {
            if (!attr_ptr) continue;
            auto* a = attr_ptr.get();
            if (a->key == "kernel_shape" && a->list && a->list->i.size() >= 2) {
                kernelY = a->list->i[0];
                kernelX = a->list->i[1];
            } else if (a->key == "group") {
                group = a->i;
            }
        }
        if (kernelY != 3 || kernelX != 5 || group != 3) {
            // Atomic EE: Extra(Conv, 3×3, group=3, unsharp) — standalone, no chain
            if (kernelY == 3 && kernelX == 3 && group == 3) {
                std::vector<float> u = {float(mW),float(mH),0.5f,0.01f, 0,0,0,0};
                ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
                auto* ex = new MNN::ExtraT(); ex->type = "isp.ee";
                buildCommonAttrs(ex, mW, mH, u);
                addNamedFloats(ex, "ee", {0.5f, 0.01f});
                setEngine(ex);
                addSpirv(ex, "isp.ee");
                ops[i]->main.value = ex;
                return true;
            }
            return false;
        }
        // Exclude FCS: search for UnaryOp (Abs) that consumes this Extra's output
        int convOut = ops[i]->outputIndexes[0];
        for (int s = i + 1; s < std::min((int)ops.size(), i + 25); s++) {
            if (!ops[s]) continue;
            if (ops[s]->type == MNN::OpType_UnaryOp) {
                for (int inIdx : ops[s]->inputIndexes)
                    if (inIdx == convOut) return false;  // FCS has Abs(edge)
            }
        }
        // Absorb chain after Extra — consume all consumers up to graph output
        int last = i;
        for (int cur = i; cur + 1 < (int)ops.size() && ops[cur + 1]; ) {
            // Skip forward past skippable ops
            int n = cur + 1;
            while (n < (int)ops.size() && ops[n] &&
                   (ops[n]->type == MNN::OpType_Const || ops[n]->type == MNN::OpType_ConvertTensor ||
                    ops[n]->type == MNN::OpType_Identity || ops[n]->type == MNN::OpType_Reshape ||
                    ops[n]->type == MNN::OpType_Squeeze || ops[n]->type == MNN::OpType_Unsqueeze)) {
                cur = n; n = cur + 1;
            }
            if (n >= (int)ops.size() || !ops[n]) break;
            // Consume if the op's inputs include THIS Extra's output tensor
            bool consumesExtra = false;
            for (int inIdx : ops[n]->inputIndexes)
                if (inIdx == convOut || inIdx == ops[last]->outputIndexes[0]) { consumesExtra = true; break; }
            if (!consumesExtra) break;
            // Stop at ops that indicate a DIFFERENT pipeline stage
            auto nt = ops[n]->type;
            if (nt == MNN::OpType_Pooling) break;
            if (nt == MNN::OpType_Extra) {
                // Allow Clip Extras (part of EE chain), break on other Extras
                auto* ex2 = ops[n]->main.AsExtra();
                if (!ex2 || ex2->type != "Clip") break;
                // Consume Clip Extra (part of EE sequence)
                last = n; cur = n; continue;
            }
            last = n; cur = n;
        }
        if (last == i) return false;
        // Convert to isp.ee
        std::vector<float> u = {float(mW),float(mH),0.5f,0.01f, 0,0,0,0};
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ee = new MNN::ExtraT(); ee->type = "isp.ee";
        buildCommonAttrs(ee, mW, mH, u);
        addNamedFloats(ee, "ee", {0.5f, 0.01f});
        setEngine(ee);
        addSpirv(ee, "isp.ee");
        ops[i]->main.value = ee;
        if (last != i) {
            ops[i]->outputIndexes[0] = ops[last]->outputIndexes[0];
            for (int r = i+1; r <= last; r++) ops[r].reset();
        }
        VLOG(2) << "[P1] Rust EE (Extra) at " << i << " (chain to " << last << ")";
        i = last;
        return true;
    }

    // Rust DisplayBlock: detect Mul(scale≈1) → isp.display identity gamma
    bool tryRustDisplay(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        // Check if one input is Const(≈1.0) — the identity scale
        bool hasIdentityScale = false;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int j = 0; j < (int)ops.size(); j++) {
                if (!ops[j] || ops[j]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[j]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[j]->main.AsBlob();
                        if (blb && !blb->float32s.empty() && std::abs(blb->float32s[0] - 1.0f) < 0.01f) {
                            hasIdentityScale = true;
                        }
                        break;
                    }
                }
                if (hasIdentityScale) break;
            }
            if (hasIdentityScale) break;
        }
        if (!hasIdentityScale) return false;
        std::vector<float> u = {float(mW),float(mH),2.2f,0.0f, 0,0,0,0};
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.display";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "display", {2.2f, 0.0f});
        setEngine(ex);
        addSpirv(ex, "isp.display");
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] Rust Display at " << i;
        return true;
    }

// ── New Post-Processing ISP Op Rules ──

    // R13: Vignetting — Mul(input, gain_map)
    // Detect: Mul with second input being a Constant (gain_map)
    bool tryVignetting(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        
        // Check if one input is a Constant (gain_map)
        bool hasGainMap = false;
        int gainMapIdx = -1;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int j = 0; j < (int)ops.size(); j++) {
                if (!ops[j] || ops[j]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[j]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[j]->main.AsBlob();
                        if (blb && blb->float32s.size() > 100) {  // gain_map is large
                            hasGainMap = true;
                            gainMapIdx = j;
                        }
                    }
                }
            }
        }
        if (!hasGainMap || gainMapIdx < 0) return false;
        
        // Extract gain map data
        auto* blb = ops[gainMapIdx]->main.AsBlob();
        std::vector<float> gainMap(blb->float32s.begin(), blb->float32s.end());
        
        // Create isp.vignetting Extra op
        ops[i]->type = MNN::OpType_Extra;
        ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT();
        ex->type = "isp.vignetting";
        std::vector<float> u = {float(mW), float(mH), 1.0f, 0, 0, 0, 0, 0};
        buildCommonAttrs(ex, mW, mH, u);
        // Store gain_map as tensor attribute
        auto* gainAttr = new MNN::AttributeT();
        gainAttr->key = "gain_map";
        gainAttr->tensor.reset(new MNN::BlobT());
        gainAttr->tensor->dataType = MNN::DataType_DT_FLOAT;
        gainAttr->tensor->float32s = gainMap;
        ex->attr.push_back(std::unique_ptr<MNN::AttributeT>(gainAttr));
        setEngine(ex);
        addSpirv(ex, "isp.vignetting");
        ops[i]->main.value = ex;
        
        // Remove the gain_map Const op
        ops[gainMapIdx].reset();
        
        VLOG(2) << "[P1] R13: vignetting at " << i << " gain_map_size=" << gainMap.size();
        return true;
    }
    
    // R14: Auto Contrast — Add → Sub → Mul → Add chain
    // Detect: shadow_lift + contrast stretch
    bool tryAutoContrast(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        // Look for Add(input, lift) → Sub(x, 0.5) → Mul(x, contrast) → Add(x, 0.5)
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_ADD)) return false;
        
        float lift = 0.0f, contrast = 1.0f;
        int add1Idx = i;
        
        // Check if first Add has a Const input (lift value)
        for (int inIdx : ops[i]->inputIndexes) {
            for (int j = 0; j < (int)ops.size(); j++) {
                if (!ops[j] || ops[j]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[j]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[j]->main.AsBlob();
                        if (blb && !blb->float32s.empty()) {
                            lift = blb->float32s[0];
                        }
                    }
                }
            }
        }
        
        // Look for Sub(x, 0.5) after Add
        int subIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
            if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
            if (isBinaryType(ops[j].get(), MNN::BinaryOpOperation_SUB)) {
                if (isChain(ops[i].get(), ops[j].get())) {
                    subIdx = j;
                }
            }
            break;
        }
        if (subIdx < 0) return false;
        
        // Look for Mul(x, contrast) after Sub
        int mulIdx = -1;
        for (int j = subIdx + 1; j < std::min((int)ops.size(), subIdx + 3); j++) {
            if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
            if (isBinaryType(ops[j].get(), MNN::BinaryOpOperation_MUL)) {
                if (isChain(ops[subIdx].get(), ops[j].get())) {
                    mulIdx = j;
                    // Extract contrast from Const input
                    for (int inIdx : ops[j]->inputIndexes) {
                        for (int k = 0; k < (int)ops.size(); k++) {
                            if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                            for (int outIdx : ops[k]->outputIndexes) {
                                if (outIdx == inIdx) {
                                    auto* blb = ops[k]->main.AsBlob();
                                    if (blb && !blb->float32s.empty()) {
                                        contrast = blb->float32s[0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
        if (mulIdx < 0) return false;
        
        // Create isp.auto_contrast Extra op
        ops[mulIdx]->type = MNN::OpType_Extra;
        ops[mulIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT();
        ex->type = "isp.auto_contrast";
        std::vector<float> u = {float(mW), float(mH), lift, contrast, 0, 0, 0, 0};
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "auto_contrast", {lift, contrast});
        setEngine(ex);
        addSpirv(ex, "isp.auto_contrast");
        ops[mulIdx]->main.value = ex;
        
        // Remove consumed ops
        ops[i].reset();      // Add
        ops[subIdx].reset(); // Sub
        
        VLOG(2) << "[P1] R14: auto_contrast at " << mulIdx << " lift=" << lift << " contrast=" << contrast;
        i = mulIdx;
        return true;
    }



    // ══════════════════════════════════════════════════════════════════════
    //  NEW ISP BLOCKS - Missing optimization rules
    // ═════════════════════════════════════════════════════════════════════

    // DPC (Defective Pixel Correction) - median filter pattern
    bool tryDpc(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Pooling) return false;
        auto* pool = ops[i]->main.AsPool();
        if (!pool || pool->kernelX != 3 || pool->kernelY != 3) return false;
        if (pool->type != MNN::PoolType_MAXPOOL) return false;

        int subIdx = -1, mulIdx = -1, addIdx = -1, clipIdx = -1;
        for (int k = i + 1; k < std::min((int)ops.size(), i + 6); k++) {
            if (!ops[k]) continue;
            if (ops[k]->type == MNN::OpType_BinaryOp) {
                auto* bin = ops[k]->main.AsBinaryOp();
                if (bin->opType == MNN::BinaryOpOperation_SUB && subIdx < 0 && isChain(ops[i].get(), ops[k].get())) subIdx = k;
                else if (bin->opType == MNN::BinaryOpOperation_MUL && mulIdx < 0 && (subIdx < 0 || isChain(ops[subIdx].get(), ops[k].get()))) mulIdx = k;
                else if (bin->opType == MNN::BinaryOpOperation_ADD && addIdx < 0 && (mulIdx < 0 || isChain(ops[mulIdx].get(), ops[k].get()))) addIdx = k;
            } else if (ops[k]->type == MNN::OpType_ReLU || ops[k]->type == MNN::OpType_ReLU6) {
                clipIdx = k;
            }
        }

        if (subIdx >= 0 && mulIdx >= 0 && addIdx >= 0) {
            int lastIdx = clipIdx >= 0 ? clipIdx : addIdx;
            ops[lastIdx]->type = MNN::OpType_Extra;
            ops[lastIdx]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.dpc"; ex->engine = "MNN";
            std::vector<float> u = {float(mW), float(mH), 3.0f, 3.0f, 0.5f};
            buildCommonAttrs(ex, mW, mH, u);
            addNamedFloats(ex, "dpc", {3.0f, 3.0f, 0.5f});
            setEngine(ex); addSpirv(ex, "isp.dpc");
            ops[lastIdx]->main.value = ex;

            if (subIdx >= 0) ops[subIdx].reset();
            if (mulIdx >= 0 && mulIdx != lastIdx) ops[mulIdx].reset();
            if (clipIdx >= 0 && clipIdx != lastIdx) ops[clipIdx].reset();

            VLOG(2) << "[P1] DPC: median filter at " << i; i = lastIdx; return true;
        }
        return false;
    }

    // Gaussian Denoise
    bool tryGaussianDenoise(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        auto* conv = ops[i]->main.AsConvolution2D();
        if (!conv || !conv->common) return false;
        if (!((conv->common->kernelX == 3 && conv->common->kernelY == 3) || (conv->common->kernelX == 5 && conv->common->kernelY == 5))) return false;
        if (conv->common->strideX != 1 || conv->common->strideY != 1) return false;

        int addIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
            if (!isBinaryType(ops[j].get(), MNN::BinaryOpOperation_ADD)) continue;
            if (isChain(ops[i].get(), ops[j].get())) { addIdx = j; break; }
        }
        if (addIdx < 0) return false;

        float blendAlpha = 0.5f;
        for (int inIdx : ops[addIdx]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == addIdx) {
                        auto* blb = ops[k]->main.AsBlob();
                        if (blb && !blb->float32s.empty()) blendAlpha = blb->float32s[0];
                    }
                }
            }
        }

        ops[addIdx]->type = MNN::OpType_Extra; ops[addIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.denoise"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), blendAlpha, 3.0f};
        buildCommonAttrs(ex, mW, mH, u); addNamedFloats(ex, "denoise", {blendAlpha, 3.0f});
        setEngine(ex); addSpirv(ex, "isp.denoise"); ops[addIdx]->main.value = ex;
        ops[i].reset(); VLOG(2) << "[P1] Denoise: Gaussian blur at " << i << " alpha=" << blendAlpha; i = addIdx; return true;
    }

    // LSC (Lens Shading Correction)
    bool tryLsc(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        int gainMapIdx = -1;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[k]->main.AsBlob();
                        if (blb && blb->dims.size() >= 2) {
                            int h = blb->dims[blb->dims.size() - 2];
                            int w = blb->dims[blb->dims.size() - 1];
                            if (h > 1 && w > 1 && h == mH && w == mW) { gainMapIdx = k; break; }
                        }
                    }
                }
                if (gainMapIdx >= 0) break;
            }
            if (gainMapIdx < 0) return false;

            ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.lsc"; ex->engine = "MNN";
            std::vector<float> u = {float(mW), float(mH)}; buildCommonAttrs(ex, mW, mH, u);
            auto* blb = ops[gainMapIdx]->main.AsBlob();
            if (blb && !blb->float32s.empty()) addNamedFloats(ex, "gain_map", blb->float32s);
            setEngine(ex); addSpirv(ex, "isp.lsc"); ops[i]->main.value = ex;
            VLOG(2) << "[P1] LSC: radial gain map at " << i; return true;
        }
        return false;
    }

    // AWB (Auto White Balance)
    bool tryAwb(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        float gains[3] = {1,1,1}; int gainConstIdx = -1;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[k]->main.AsBlob();
                        if (blb && blb->float32s.size() >= 3) { for (int c=0;c<3;c++) gains[c]=blb->float32s[c]; gainConstIdx=k; break; }
                    }
                } if (gainConstIdx >= 0) break;
            } if (gainConstIdx < 0) return false;

            int addIdx = -1;
            for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
                if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
                if (isBinaryType(ops[j].get(), MNN::BinaryOpOperation_ADD) && isChain(ops[i].get(), ops[j].get())) { addIdx = j; break; }
            } if (addIdx < 0) return false;

            float offsets[3] = {0,0,0};
            for (int inIdx : ops[addIdx]->inputIndexes) {
                for (int k = 0; k < (int)ops.size(); k++) {
                    if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                    for (int outIdx : ops[k]->outputIndexes) {
                        if (outIdx == inIdx) {
                            auto* blb = ops[k]->main.AsBlob();
                            if (blb && blb->float32s.size() >= 3) { for(int c=0;c<3;c++) offsets[c]=blb->float32s[c]; } break;
                        }
                    }
                }
            }

            ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.awb"; ex->engine = "MNN";
            std::vector<float> u = {float(mW), float(mH), gains[0], gains[1], gains[2], offsets[0], offsets[1], offsets[2]};
            buildCommonAttrs(ex, mW, mH, u); addNamedFloats(ex, "awb", {gains[0], gains[1], gains[2], offsets[0], offsets[1], offsets[2]});
            setEngine(ex); addSpirv(ex, "isp.awb"); ops[i]->main.value = ex;
            if (gainConstIdx >= 0) ops[gainConstIdx].reset(); if (addIdx >= 0) ops[addIdx].reset();
            VLOG(2) << "[P1] AWB: channel gains at " << i; return true;
        }
        return false;
    }

    // AE (Auto Exposure)
    bool tryAe(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        float gain = 1.0f; int gainConstIdx = -1;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) { auto* blb = ops[k]->main.AsBlob(); if (blb && blb->float32s.size() == 1) { gain = blb->float32s[0]; gainConstIdx = k; break; } }
                } if (gainConstIdx >= 0) break;
            } if (gainConstIdx < 0) return false;

            float offset = 0.0f; int addIdx = -1;
            for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
                if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
                if (isBinaryType(ops[j].get(), MNN::BinaryOpOperation_ADD) && isChain(ops[i].get(), ops[j].get())) {
                    for (int inIdx : ops[j]->inputIndexes) {
                        for (int k = 0; k < (int)ops.size(); k++) {
                            if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                            for (int outIdx : ops[k]->outputIndexes) {
                                if (outIdx == inIdx) { auto* blb = ops[k]->main.AsBlob(); if (blb && blb->float32s.size() == 1) { offset = blb->float32s[0]; addIdx = j; break; } }
                            } if (addIdx >= 0) break;
                        }
                    } if (addIdx >= 0) break;
                }

                ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
                auto* ex = new MNN::ExtraT(); ex->type = "isp.ae"; ex->engine = "MNN";
                std::vector<float> u = {float(mW), float(mH), gain, offset};
                buildCommonAttrs(ex, mW, mH, u); addNamedFloats(ex, "ae", {gain, offset}); setEngine(ex); addSpirv(ex, "isp.ae");
                ops[i]->main.value = ex; if (gainConstIdx >= 0) ops[gainConstIdx].reset(); if (addIdx >= 0) ops[addIdx].reset();
                VLOG(2) << "[P1] AE: global gain=" << gain << " offset=" << offset << " at " << i; return true;
            }
        }
        return false;
    }

    // Tone Mapping
    bool tryTone(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        auto* bin = ops[i]->main.AsBinaryOp(); if (!bin || bin->opType != MNN::BinaryOpOperation_POW) return false;
        float gamma = 2.2f;
        // Try to read gamma from the const input
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) { auto* blb = ops[k]->main.AsBlob(); if (blb && !blb->float32s.empty()) gamma = blb->float32s[0]; }
                }
            }
        }
        float contrast = 1.0f; int mulIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
            if (isBinaryType(ops[j].get(), MNN::BinaryOpOperation_MUL) && isChain(ops[i].get(), ops[j].get())) {
                for (int inIdx : ops[j]->inputIndexes) {
                    for (int k = 0; k < (int)ops.size(); k++) {
                        if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                        for (int outIdx : ops[k]->outputIndexes) {
                            if (outIdx == inIdx) { auto* blb = ops[k]->main.AsBlob(); if (blb && blb->float32s.size() == 1) { contrast = blb->float32s[0]; mulIdx = j; break; } }
                        }
                    } if (mulIdx >= 0) break;
                } if (mulIdx >= 0) break;
            }

            int unsharpIdx = -1;
            if (mulIdx >= 0) {
                for (int j = mulIdx + 1; j < std::min((int)ops.size(), mulIdx + 4); j++) {
                    if (!ops[j] || ops[j]->type != MNN::OpType_Convolution) continue;
                    auto* conv = ops[j]->main.AsConvolution2D();
                    if (conv && conv->common && conv->common->kernelX == 3 && conv->common->kernelY == 3) { unsharpIdx = j; break; }
                }
            }

            int replaceIdx = unsharpIdx >= 0 ? unsharpIdx : (mulIdx >= 0 ? mulIdx : i);
            ops[replaceIdx]->type = MNN::OpType_Extra; ops[replaceIdx]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.tone"; ex->engine = "MNN";
            std::vector<float> u = {float(mW), float(mH), gamma, contrast, 0, 0, 0, 0};
            buildCommonAttrs(ex, mW, mH, u); addNamedFloats(ex, "tone", {gamma, contrast}); setEngine(ex); addSpirv(ex, "isp.tone");
            ops[replaceIdx]->main.value = ex;
            if (mulIdx >= 0 && mulIdx != replaceIdx) ops[mulIdx].reset();
            if (unsharpIdx >= 0 && unsharpIdx != replaceIdx) ops[unsharpIdx].reset();
            VLOG(2) << "[P1] Tone: gamma=" << gamma << " contrast=" << contrast << " at " << i; i = replaceIdx; return true;
        }
        return false;
    }

    // Gamma correction
    bool tryGamma(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        auto* bin = ops[i]->main.AsBinaryOp(); if (!bin || bin->opType != MNN::BinaryOpOperation_POW) return false;
        float gamma = 2.2f;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) { auto* blb = ops[k]->main.AsBlob(); if (blb && !blb->float32s.empty()) gamma = blb->float32s[0]; }
                }
            }
        }
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.gamma"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), gamma}; buildCommonAttrs(ex, mW, mH, u); addNamedFloats(ex, "gamma", {gamma}); setEngine(ex); addSpirv(ex, "isp.gamma");
        ops[i]->main.value = ex; VLOG(2) << "[P1] Gamma: " << gamma << " at " << i; return true;
    }

    // Calibration Stats
    bool tryCalibStats(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Reduction) return false;
        auto* red = ops[i]->main.AsReductionParam(); if (!red) return false;
        if (red->operation != MNN::ReductionType_MEAN && red->operation != MNN::ReductionType_MIN && red->operation != MNN::ReductionType_MAX) return false;
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.calib_stats"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), (float)red->operation}; buildCommonAttrs(ex, mW, mH, u); setEngine(ex); addSpirv(ex, "isp.calib_stats");
        ops[i]->main.value = ex; VLOG(2) << "[P1] CalibStats: op=" << (int)red->operation << " at " << i; return true;
    }

    // IspController Stats
    bool tryIspControllerStats(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Reduction) return false;
        auto* red = ops[i]->main.AsReductionParam(); if (!red) return false;
        if (red->operation != MNN::ReductionType_MEAN) return false;
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.ispc_stats"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH)}; buildCommonAttrs(ex, mW, mH, u); setEngine(ex); addSpirv(ex, "isp.ispc_stats");
        ops[i]->main.value = ex; VLOG(2) << "[P1] IspControllerStats at " << i; return true;
    }

    // AF Focus
    bool tryAfFocus(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        auto* conv = ops[i]->main.AsConvolution2D(); if (!conv || !conv->common || conv->common->kernelX != 3 || conv->common->kernelY != 3) return false;
        float expectedSobel[9] = {-1,0,1, -2,0,2, -1,0,1}; bool isSobel = false;
        if (conv->weight.size() >= 9) { isSobel = true; for (int k=0;k<9;k++) if (fabs(conv->weight[k]-expectedSobel[k])>0.1f){isSobel=false;break;} } if (!isSobel) return false;
        int mulIdx=-1, redIdx=-1, powIdx=-1, red2Idx=-1;
        for (int j=i+1;j<std::min((int)ops.size(),i+6);j++) {
            if(!ops[j])continue;
            if(ops[j]->type==MNN::OpType_BinaryOp&&mulIdx<0){ if(isBinaryType(ops[j].get(),MNN::BinaryOpOperation_MUL)&&isChain(ops[i].get(),ops[j].get())) mulIdx=j; }
            else if(ops[j]->type==MNN::OpType_Reduction&&redIdx<0&&mulIdx>=0){ if(isChain(ops[mulIdx].get(),ops[j].get())) redIdx=j; }
            else if(ops[j]->type==MNN::OpType_UnaryOp&&powIdx<0&&redIdx>=0){ auto* bin=ops[j]->main.AsBinaryOp(); if(bin&&bin->opType==MNN::BinaryOpOperation_POW) powIdx=j; }
            else if(ops[j]->type==MNN::OpType_Reduction&&red2Idx<0&&powIdx>=0){ if(isChain(ops[powIdx].get(),ops[j].get())) red2Idx=j; }
        } if(red2Idx<0) return false;
        ops[red2Idx]->type=MNN::OpType_Extra; ops[red2Idx]->main.type=MNN::OpParameter_Extra;
        auto* ex=new MNN::ExtraT(); ex->type="isp.af_focus"; ex->engine="MNN";
        std::vector<float> u={float(mW),float(mH)}; buildCommonAttrs(ex,mW,mH,u); setEngine(ex); addSpirv(ex,"isp.af_focus");
        ops[red2Idx]->main.value=ex; ops[i].reset(); if(mulIdx>=0)ops[mulIdx].reset(); if(redIdx>=0)ops[redIdx].reset(); if(powIdx>=0)ops[powIdx].reset();
        VLOG(2)<<"[P1] AF Focus at "<<i; i=red2Idx; return true;
    }

    // EIS Gyro
    bool tryEisGyro(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if(!ops[i]||ops[i]->type!=MNN::OpType_Extra)return false;
        auto* ex=ops[i]->main.AsExtra(); if(!ex||ex->type!="isp.warp")return false;
        bool hasGyro=false; for(auto& attr:ex->attr){ if(attr&&(attr->key=="gyro_x"||attr->key=="gyro_y"||attr->key=="gyro_z")){hasGyro=true;break;} }
        if(!hasGyro)return false; ex->type="isp.eis_gyro"; addSpirv(ex,"isp.eis_gyro");
        VLOG(2)<<"[P1] EIS Gyro at "<<i; return true;
    }

};

// ═══════════════════════════════════════════════════════════════════
//  Pass 2: ISP Extra chain → fused Extra
// ═══════════════════════════════════════════════════════════════════

class Pass2_FuseExtra : public PostConverter {
public:
    static Pass2_FuseExtra* instance() {
        static Pass2_FuseExtra p;
        return &p;
    }

    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;

        // Extract dimensions from the first Extra op's output_shape
        for (auto& op : ops) {
            if (op && op->type == MNN::OpType_Extra && op->main.AsExtra()) {
                auto* ex = op->main.AsExtra();
                for (auto& attr : ex->attr) {
                    if (attr && attr->key == "output_shape" && attr->tensor &&
                        attr->tensor->int32s.size() >= 4) {
                        mH = attr->tensor->int32s[2];
                        mW = attr->tensor->int32s[3];
                        break;
                    }
                }
                if (mW > 0 && mH > 0) break;
            }
        }
        if (mW == 0 || mH == 0) {
            mW = 1920; mH = 1080;
        }

        // Walk in pipeline order: after Pass1 the ops form a linear chain
        // Input → Extra(unpack) → Extra(demosaic) → Extra(fcs) → ...
        // Collect consecutive Extra ops and fuse adjacent valid pairs.
        bool any = true;
        while (any) {
            any = false;

            // Collect pipeline Extras in tensor chain order from Input
            std::vector<int> extras;
            {
                int cur = -1;
                for (auto& op : ops)
                    if (op && op->type == MNN::OpType_Input && !op->outputIndexes.empty())
                        { cur = op->outputIndexes[0]; break; }
                // Try chain order first (tracing tensor from Input)
                if (cur >= 0) {
                    while (cur >= 0) {
                        bool found = false;
                        for (int j = 0; j < (int)ops.size(); j++) {
                            if (!ops[j] || ops[j]->type != MNN::OpType_Extra) continue;
                            for (int inIdx : ops[j]->inputIndexes)
                                if (traceTensor(inIdx, ops) == cur) {
                                    extras.push_back(j);
                                    cur = ops[j]->outputIndexes.empty() ? -1 : ops[j]->outputIndexes[0];
                                    found = true; break;
                                }
                            if (found) break;
                        }
                        if (!found) break;
                    }
                }
                // If chain tracing failed, fall back to scanning all Extras
                // and linking them by their tensor positions
                if (extras.size() < 2) {
                    extras.clear();
                    for (int j = 0; j < (int)ops.size(); j++) {
                        if (ops[j] && ops[j]->type == MNN::OpType_Extra) {
                            extras.push_back(j);
                        }
                    }
                    VLOG(1) << "[P2] fallback scan found " << extras.size() << " Extras";
                }
            }
            if (extras.size() < 2) break;

            // R10: unpack_blc + demosaic_ccm → unpack_demosaic
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k], j = extras[k+1];
                if (isExtraOfType(ops[i].get(), "isp.unpack_blc") &&
                    isExtraOfType(ops[j].get(), "isp.demosaic_ccm") &&
                    matchUnpackDemosaic(ops, i, j)) {
                    any = true; break;
                }
            }
            if (any) continue;

            // R10b: unpack_blc + isp.demosaic(algorithm=binning) → unpack_demosaic
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k], j = extras[k+1];
                if (isExtraOfType(ops[i].get(), "isp.unpack_blc") &&
                    isExtraOfTypeWithAlgo(ops[j].get(), "isp.demosaic", "binning") &&
                    matchUnpackDemosaicFromUnified(ops, i, j)) {
                    any = true; break;
                }
            }
            if (any) continue;

            // R8: fcs + display → fcs_display (must fire before R9 to avoid ee_ldci
            //     blocking the fcs+display adjacency)
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k], j = extras[k+1];
                if (isExtraOfType(ops[i].get(), "isp.fcs") &&
                    isExtraOfType(ops[j].get(), "isp.display") &&
                    matchFcsDisplay(ops, i, j)) {
                    any = true; break;
                }
            }
            if (any) continue;

            // R9: ee + ldci → ee_ldci
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k], j = extras[k+1];
                if (!ops[i] || !ops[j]) continue;
                if ((isExtraOfType(ops[i].get(), "isp.ee") && isExtraOfType(ops[j].get(), "isp.ldci")) ||
                    (isExtraOfType(ops[i].get(), "isp.ldci") && isExtraOfType(ops[j].get(), "isp.ee"))) {
                    if (matchEeLdci(ops, i, j)) { any = true; break; }
                }
            }
            if (any) continue;

            // R11: unpack_demosaic + fcs_display → unpack_demosaic (fuse display gamma)
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k], j = extras[k+1];
                if ((isExtraOfType(ops[i].get(), "isp.unpack_demosaic") &&
                     isExtraOfType(ops[j].get(), "isp.fcs_display")) &&
                    matchUnpackDisplay(ops, i, j)) {
                    any = true; break;
                }
            }
            if (any) continue;

            // R12: unpack_demosaic + fcs → unpack_demosaic (fuse FCS into unpack shader)
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k], j = extras[k+1];
                if (matchUnpackFcs(ops, i, j)) {
                    any = true; break;
                }
            }
            if (any) continue;

            // R11b: unpack_demosaic + ... + display → unpack_demosaic (fuse display gamma)
            // After R12 and R9, we may have unpack_demosaic_fcs, ee_ldci, display.
            // This rule skips over cosmetic intermediates (ee_ldci, ee, ldci, fcs)
            // to absorb display gamma directly into unpack_demosaic.
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k];
                if (!isExtraOfType(ops[i].get(), "isp.unpack_demosaic")) continue;
                // Find display somewhere after i, skipping only cosmetic extras
                for (size_t kk = k + 1; kk < extras.size(); kk++) {
                    int mid = extras[kk];
                    if (isExtraOfType(ops[mid].get(), "isp.display")) {
                        if (matchUnpackDisplayDirect(ops, i, mid)) {
                            any = true; break;
                        }
                    }
                    // Allow cosmetic intermediates
                    if (!isExtraOfType(ops[mid].get(), "isp.ee_ldci") &&
                        !isExtraOfType(ops[mid].get(), "isp.ee") &&
                        !isExtraOfType(ops[mid].get(), "isp.ldci") &&
                        !isExtraOfType(ops[mid].get(), "isp.fcs")) {
                        break;  // non-cosmetic op in between → stop
                    }
                }
                if (any) break;
            }
            if (any) continue;

            break;
        }

        ops.erase(std::remove_if(ops.begin(), ops.end(),
                  [](const std::unique_ptr<OpT>& o) { return !o; }), ops.end());
        if (changed) VLOG(1) << "[P2] Fusion complete: " << ops.size() << " ops";
        return changed;
    }

private:
    mutable int mW = 1920, mH = 1080;

    bool merge2(MNN::NetT* net, const std::vector<int>& idx,
                const char* fusedType, const std::vector<float>& uniforms) {
        auto* first = net->oplists[idx[0]].get();
        auto* last  = net->oplists[idx.back()].get();

        first->main.AsExtra()->type = fusedType;
        first->main.AsExtra()->attr.clear();  // drop old attrs
        buildCommonAttrs(first->main.AsExtra(), mW, mH, uniforms);
        setEngine(first->main.AsExtra());
        addSpirv(first->main.AsExtra(), fusedType);
        first->outputIndexes[0] = last->outputIndexes[0];

        for (size_t k = 1; k < idx.size(); k++)
            net->oplists[idx[k]].reset();
        return true;
    }

    // R8: isp.fcs + isp.display → isp.fcs_display (pair at indices i,j)
    bool matchFcsDisplay(std::vector<std::unique_ptr<OpT>>& ops, int i, int j) const {
        VLOG(1) << "[P2] R8: FcsDisplay at " << i << "+" << j;
        auto* fcs = ops[i]->main.AsExtra();
        int W, H;
        getExtraDims(ops[i], W, H);
        // Read fcs strength from named attrs
        auto fcsVals = getNamedFloats(fcs, "fcs");
        float str = (fcsVals.size() >= 1) ? fcsVals[0] : 1.0f;
        std::vector<float> u = {float(W),float(H), str,0, 2.2f,0, 0,0,0};
        ops[i]->main.AsExtra()->type = "isp.fcs_display";
        ops[i]->main.AsExtra()->attr.clear();
        buildCommonAttrs(ops[i]->main.AsExtra(), W, H, u);
        addNamedFloats(ops[i]->main.AsExtra(), "fcs",     {str, 0.0f});
        addNamedFloats(ops[i]->main.AsExtra(), "display", {2.2f, 0.0f});
        setEngine(ops[i]->main.AsExtra());
        addSpirv(ops[i]->main.AsExtra(), "isp.fcs_display");
        ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
        ops[j].reset();
        return true;
    }

    // R9: isp.ee + isp.ldci → isp.ee_ldci (pair at indices i,j, either order)
    bool matchEeLdci(std::vector<std::unique_ptr<OpT>>& ops, int i, int j) const {
        // Determine order: ee→ldci or ldci→ee
        bool ldciFirst = isExtraOfType(ops[i].get(), "isp.ldci");
        int keepIdx = ldciFirst ? j : i;  // keep the FIRST in chain order (ee)
        int resetIdx = ldciFirst ? i : j; // reset the SECOND
        
        std::string order = ldciFirst ? "ldci→ee" : "ee→ldci";
        VLOG(1) << "[P2] R9: EeLdci at " << i << "+" << j << " (" << order << ")";
        int W, H;
        getExtraDims(ops[keepIdx], W, H);
        if (W <= 0 || H <= 0) getExtraDims(ops[resetIdx], W, H);
        if (W <= 0 || H <= 0) { W = 1920; H = 1080; }
        std::vector<float> u = {float(W),float(H), 0.5f,0.01f, 0.5f,1.0f, 0,0};
        // Read named params from source ops
        auto* eeEx = ops[ldciFirst ? j : i]->main.AsExtra();
        auto* ldciEx = ops[ldciFirst ? i : j]->main.AsExtra();
        auto eeVals = getNamedFloats(eeEx, "ee");
        auto ldciVals = getNamedFloats(ldciEx, "ldci");
        if (eeVals.size() >= 2) { u[2] = eeVals[0]; u[3] = eeVals[1]; }
        if (ldciVals.size() >= 2) { u[4] = ldciVals[0]; u[5] = ldciVals[1]; }
        ops[keepIdx]->main.AsExtra()->type = "isp.ee_ldci";
        ops[keepIdx]->main.AsExtra()->attr.clear();
        buildCommonAttrs(ops[keepIdx]->main.AsExtra(), W, H, u);
        addNamedFloats(ops[keepIdx]->main.AsExtra(), "ee",   {u[2], u[3]});
        addNamedFloats(ops[keepIdx]->main.AsExtra(), "ldci", {u[4], u[5]});
        setEngine(ops[keepIdx]->main.AsExtra());
        addSpirv(ops[keepIdx]->main.AsExtra(), "isp.ee_ldci");
        ops[keepIdx]->outputIndexes[0] = ops[resetIdx]->outputIndexes[0];
        ops[resetIdx].reset();
        return true;
    }

    // R10: isp.unpack_blc + isp.demosaic_ccm → isp.unpack_demosaic
    // Extract W,H from an existing Extra op's output_shape attribute
    void getExtraDims(const std::unique_ptr<OpT>& op, int& W, int& H) const {
        W = 1920; H = 1080;
        if (!op || op->type != MNN::OpType_Extra) return;
        auto* ex = op->main.AsExtra();
        if (!ex) return;
        for (auto& attr : ex->attr) {
            if (attr && attr->key == "output_shape" && attr->tensor &&
                attr->tensor->int32s.size() >= 4) {
                H = attr->tensor->int32s[2];
                W = attr->tensor->int32s[3];
                break;
            }
        }
    }

    // R10: isp.unpack_blc + isp.demosaic_ccm → isp.unpack_demosaic (pair at i,j)
    bool matchUnpackDemosaic(std::vector<std::unique_ptr<OpT>>& ops, int i, int j) const {
        VLOG(1) << "[P2] R10: UnpackDemosaic at " << i << "+" << j;
        int W, H;
        getExtraDims(ops[j], W, H);   // demosaic dims (output=FHD)
        int inpW = W*2, inpH = H*2;    // input dims (Bayer=4K)

        // Build const buffer for unpack_demosaic:
        // [dims4, smax, blc4, wb4, ccm9, fcs2, bayer_pat, gamma]
        std::vector<float> u = {float(W),float(H), float(inpW),float(inpH), 1023,
                                0,0,0,0, 1,1,1,1,
                                1,0,0, 0,1,0, 0,0,1,
                                1.0f, 0.0f,  // fcs_str=1.0, fcs_off=0.0 (default)
                                0,0};  // bayer_pattern=0(RGGB), gamma=0

        // Read blc/wb from unpack_blc's const buffer (positions [5..12])
        auto unpackConst = getExtraConst(ops[i]);
        if (unpackConst.size() >= 13) {
            u[5] = unpackConst[5];  u[6] = unpackConst[6];
            u[7] = unpackConst[7];  u[8] = unpackConst[8];
            u[9] = unpackConst[9];  u[10] = unpackConst[10];
            u[11] = unpackConst[11]; u[12] = unpackConst[12];
        }

        // Read CCM from demosaic_ccm's const buffer (positions [5..13])
        auto ccmConst = getExtraConst(ops[j]);
        if (ccmConst.size() >= 14) {
            for (int k = 0; k < 9; k++) u[13 + k] = ccmConst[5 + k];
        }

        ops[i]->main.AsExtra()->type = "isp.unpack_demosaic";
        ops[i]->main.AsExtra()->attr.clear();
        buildCommonAttrs(ops[i]->main.AsExtra(), W, H, u);
        // Store named params for R11 fusion
        addNamedFloats(ops[i]->main.AsExtra(), "blc", {u[5],u[6],u[7],u[8]});
        addNamedFloats(ops[i]->main.AsExtra(), "wb",  {u[9],u[10],u[11],u[12]});
        addNamedFloats(ops[i]->main.AsExtra(), "ccm", {u[13],u[14],u[15],u[16],u[17],u[18],u[19],u[20],u[21]});
        setEngine(ops[i]->main.AsExtra());
        addSpirv(ops[i]->main.AsExtra(), "isp.unpack_demosaic");
        ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
        ops[j].reset();
        return true;
    }

    // Check if Extra op has a specific type AND algorithm attribute
    static bool isExtraOfTypeWithAlgo(const OpT* op, const char* type, const char* algo) {
        if (!op || op->type != MNN::OpType_Extra) return false;
        auto* e = op->main.AsExtra();
        if (!e || e->type != type) return false;
        for (auto& attr : e->attr) {
            if (attr && attr->key == "algorithm") {
                return attr->s == algo;
            }
        }
        return false;
    }

    // R10b: isp.unpack_blc + isp.demosaic(algorithm=binning) → unpack_demosaic
    // Same as R10 but for the unified isp.demosaic opset with algorithm=binning.
    bool matchUnpackDemosaicFromUnified(std::vector<std::unique_ptr<OpT>>& ops, int i, int j) const {
        VLOG(1) << "[P2] R10b: UnpackDemosaicFromUnified at " << i << "+" << j;
        int W, H;
        getExtraDims(ops[j], W, H);
        int inpW = W*2, inpH = H*2;

        std::vector<float> u = {float(W),float(H), float(inpW),float(inpH), 1023,
                                0,0,0,0, 1,1,1,1,
                                1,0,0, 0,1,0, 0,0,1,
                                1.0f, 0.0f,  // fcs_str, fcs_off
                                0,0};  // bayer_pattern=0(RGGB), gamma=0

        // Read blc/wb from unpack_blc's const buffer
        auto unpackConst = getExtraConst(ops[i]);
        if (unpackConst.size() >= 13) {
            u[5] = unpackConst[5];  u[6] = unpackConst[6];
            u[7] = unpackConst[7];  u[8] = unpackConst[8];
            u[9] = unpackConst[9];  u[10] = unpackConst[10];
            u[11] = unpackConst[11]; u[12] = unpackConst[12];
        }

        // Read CCM from demosaic's const buffer (positions [5..13])
        auto dmConst = getExtraConst(ops[j]);
        if (dmConst.size() >= 14) {
            for (int k = 0; k < 9; k++) u[13 + k] = dmConst[5 + k];
        }

        ops[i]->main.AsExtra()->type = "isp.unpack_demosaic";
        ops[i]->main.AsExtra()->attr.clear();
        buildCommonAttrs(ops[i]->main.AsExtra(), W, H, u);
        addNamedFloats(ops[i]->main.AsExtra(), "blc", {u[5],u[6],u[7],u[8]});
        addNamedFloats(ops[i]->main.AsExtra(), "wb",  {u[9],u[10],u[11],u[12]});
        addNamedFloats(ops[i]->main.AsExtra(), "ccm", {u[13],u[14],u[15],u[16],u[17],u[18],u[19],u[20],u[21]});
        setEngine(ops[i]->main.AsExtra());
        addSpirv(ops[i]->main.AsExtra(), "isp.unpack_demosaic");
        ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
        ops[j].reset();
        return true;
    }

    // R11: isp.unpack_demosaic + isp.fcs_display → unpack_demosaic (no separate display)
    // Fuses display gamma correction into the unpack_demosaic shader by writing
    // display gamma and fcs params into the const buffer at positions [22..24].
    bool matchUnpackDisplay(std::vector<std::unique_ptr<OpT>>& ops, int i, int j) const {
        VLOG(1) << "[P2] R11: UnpackDisplay at " << i << "+" << j;
        
        // Read fcs_str, fcs_off, display_gamma from fcs_display's named attrs
        auto* fcsDispEx = ops[j]->main.AsExtra();
        auto fcsVals = getNamedFloats(fcsDispEx, "fcs");
        auto dispVals = getNamedFloats(fcsDispEx, "display");
        float fcs_str = (fcsVals.size() >= 1) ? fcsVals[0] : 1.0f;
        float fcs_off = (fcsVals.size() >= 2) ? fcsVals[1] : 0.0f;
        float gamma = (dispVals.size() >= 1) ? dispVals[0] : 2.2f;
        
        // Update unpack_demosaic const buffer: positions [22..25]
        // Layout: [22]=fcs_str, [23]=fcs_off, [24]=bayer_pattern, [25]=display_gamma
        auto* ex = ops[i]->main.AsExtra();
        for (auto& attr : ex->attr) {
            if (attr && attr->key == "const" && attr->tensor &&
                attr->tensor->dataType == MNN::DataType_DT_FLOAT &&
                attr->tensor->float32s.size() >= 26) {
                attr->tensor->float32s[22] = fcs_str;
                attr->tensor->float32s[23] = fcs_off;
                // Position 24 is bayer_pattern (preserved from unpack)
                // Position 25 is display gamma (0=none, >0=apply gamma)
                attr->tensor->float32s[25] = gamma;
                break;
            }
        }
        
        VLOG(1) << "[P2] R11: UnpackDisplay at " << i << "+" << j
                << " (fcs_str=" << fcs_str << " fcs_off=" << fcs_off
                << " gamma=" << gamma << ")";
        
        // Redirect output through fcs_display's output, remove fcs_display
        ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
        ops[j].reset();
        return true;
    }

    // R12: isp.unpack_demosaic + isp.fcs → isp.unpack_demosaic (fuse FCS into unpack shader)
    // FCS is a trivial linear transform that can run in the unpack shader.
    // Eliminates one GPU dispatch and one buffer roundtrip.
    bool matchUnpackFcs(std::vector<std::unique_ptr<OpT>>& ops, int i, int j) const {
        if (!isExtraOfType(ops[i].get(), "isp.unpack_demosaic")) return false;
        if (!isExtraOfType(ops[j].get(), "isp.fcs")) return false;
        // Check if unpack_demosaic output feeds into fcs (skip intermediate Const/Mul ops)
        if (!isChainSkipCT(ops[i].get(), ops[j].get(), ops)) return false;

        // Read FCS params from named attrs
        auto* fcsEx = ops[j]->main.AsExtra();
        auto fcsVals = getNamedFloats(fcsEx, "fcs");
        float fcs_str = (fcsVals.size() >= 1) ? fcsVals[0] : 1.0f;
        float fcs_off = (fcsVals.size() >= 2) ? fcsVals[1] : 0.0f;

        // Update unpack_demosaic const buffer: write fcs params at positions [22..23]
        auto* ex = ops[i]->main.AsExtra();
        for (auto& attr : ex->attr) {
            if (attr && attr->key == "const" && attr->tensor &&
                attr->tensor->dataType == MNN::DataType_DT_FLOAT &&
                attr->tensor->float32s.size() >= 24) {
                attr->tensor->float32s[22] = fcs_str;
                attr->tensor->float32s[23] = fcs_off;
                break;
            }
        }

        VLOG(1) << "[P2] R12: UnpackFcs at " << i << "+" << j
                << " (str=" << fcs_str << " off=" << fcs_off << ")";

        // Redirect unpack_demosaic output to fcs output, remove fcs
        ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
        ops[j].reset();
        return true;
    }

    // R11b: isp.unpack_demosaic + ... + isp.display → isp.unpack_demosaic
    // Absorbs display gamma into unpack_demosaic's const buffer even when
    // cosmetic ops (ee_ldci, ee, ldci, fcs) are between them.
    // This handles the case where FCS is already fused into unpack_demosaic
    // and EE+LDCI are fused into ee_ldci, leaving:
    //   [unpack_demosaic_fcs, ee_ldci, display]
    bool matchUnpackDisplayDirect(
        std::vector<std::unique_ptr<OpT>>& ops, int i, int j) const {
        auto* dispEx = ops[j]->main.AsExtra();
        auto dispVals = getNamedFloats(dispEx, "display");
        float gamma = (dispVals.size() >= 1) ? dispVals[0] : 2.2f;
        
        // Write gamma into unpack_demosaic's const buffer at position [25]
        // Layout: [24]=bayer_pattern, [25]=display_gamma (0=none, >0=apply gamma)
        auto* ex = ops[i]->main.AsExtra();
        for (auto& attr : ex->attr) {
            if (attr && attr->key == "const" && attr->tensor &&
                attr->tensor->dataType == MNN::DataType_DT_FLOAT &&
                attr->tensor->float32s.size() >= 26) {
                attr->tensor->float32s[25] = gamma;
                break;
            }
        }
        
        VLOG(1) << "[P2] R11b: UnpackDisplayDirect at " << i << "+" << j
                << " (gamma=" << gamma << ")";
        
        // Redirect unpack_demosaic output to display output, remove display
        ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
        ops[j].reset();
        return true;
    }

};

// ═══════════════════════════════════════════════════════════════════
//  Pre-pass: Remove Identity ops (chain through them)
// ═══════════════════════════════════════════════════════════════════
// The Rust pipeline generates Identity ops for disabled features
// (lsc, tone, hook_src, hook_out). These break IspChainFusion's
// pattern matching. This pre-pass reroutes around them.

class RemoveIdentityOps : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;
        bool any = true;
        while (any) {
            any = false;
            for (int i = 0; i < (int)ops.size(); i++) {
                if (!ops[i] || ops[i]->type != MNN::OpType_Identity) continue;
                auto& id = ops[i];
                if (id->inputIndexes.size() != 1 || id->outputIndexes.size() != 1) continue;
                int inIdx = id->inputIndexes[0];
                int outIdx = id->outputIndexes[0];

                // Reroute all consumers of outIdx to inIdx
                for (auto& op : ops) {
                    if (!op || op.get() == id.get()) continue;
                    for (auto& idx : op->inputIndexes) {
                        if (idx == outIdx) {
                            idx = inIdx;
                            changed = true;
                        }
                    }
                }
                VLOG(1) << "[RemoveIdentity] Rerouted Identity at " << i
                        << " (" << inIdx << " → " << outIdx << ")";
                id.reset();
                any = true;
                break;  // restart scan
            }
        }
        if (changed) {
            ops.erase(std::remove_if(ops.begin(), ops.end(),
                [](const std::unique_ptr<OpT>& o) { return !o; }), ops.end());
        }
        return changed;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  Orchestrator: runs Pass1 → Pass2 autoregressively
// ═══════════════════════════════════════════════════════════════════

class IspChainFusion : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        VLOG(1) << "[IspFusion] === Pre-pass: Remove Identity ops ===";
        RemoveIdentityOps().onExecute(net);

        VLOG(1) << "[IspFusion] === Pass 1: Standard → ISP Extra ops ===";
        Pass1_ToExtra::instance()->onExecute(net);

        VLOG(1) << "[IspFusion] === Pass 2: MACRO-FUSION — Extra chain → fused Extra ===";
        Pass2_FuseExtra::instance()->onExecute(net);

        VLOG(1) << "[IspFusion] Complete: " << net->oplists.size() << " ops";
        // Debug: dump all op types
        for (size_t i = 0; i < net->oplists.size(); i++) {
            auto& op = net->oplists[i];
            if (op && op->type == MNN::OpType_Extra) {
                auto* ex = op->main.AsExtra();
                fprintf(stderr, "  [%zu] Extra(%s)", i, ex->type.c_str());
                for (auto& a : ex->attr) {
                    if (a && a->key == "optimized_dispatch") fprintf(stderr, " od=%d", a->b ? 1 : 0);
                }
                fprintf(stderr, "\n");
            } else if (op) {
                fprintf(stderr, "  [%zu] %s\n", i, MNN::EnumNamesOpType()[op->type]);
            }
        }
        return true;
    }
};

// ── Post-format-converter cleanup: Remove ConvertTensors between Extra ops ──
// Runs AFTER AddTensorFormatConverter to eliminate spurious
// conversions that corrupt CHW planar data.
class RemoveExtraConvertTensor : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;
        bool any = true;
        while (any) {
            any = false;
            for (int i = 0; i < (int)ops.size(); i++) {
                if (!ops[i] || ops[i]->type != MNN::OpType_ConvertTensor) continue;
                auto& ct = ops[i];
                if (ct->inputIndexes.size() != 1 || ct->outputIndexes.size() != 1) continue;
                int ctIn = ct->inputIndexes[0];
                int ctOut = ct->outputIndexes[0];
                
                // Check if ctIn comes from an Extra op
                bool fromExtra = false;
                for (auto& op : ops) {
                    if (!op || op->type != MNN::OpType_Extra) continue;
                    for (auto outIdx : op->outputIndexes) {
                        if (outIdx == ctIn) { fromExtra = true; break; }
                    }
                    if (fromExtra) break;
                }
                if (!fromExtra) continue;
                
                // Check if ctOut is consumed by an Extra op
                for (auto& op : ops) {
                    if (!op || op.get() == ct.get() || op->type != MNN::OpType_Extra) continue;
                    bool rerouted = false;
                    for (auto& inIdx : op->inputIndexes) {
                        if (inIdx == ctOut) {
                            inIdx = ctIn;  // reroute to ConvertTensor input
                            rerouted = true;
                        }
                    }
                    if (rerouted) {
                        VLOG(1) << "[RemoveExtraConvert] Rerouted ConvertTensor at " << i
                                << " (Extra " << ctIn << "→" << ctOut << ")";
                        ct.reset();
                        changed = true;
                        any = true;
                        break;
                    }
                }
                if (any) break;  // restart scan
            }
        }
        if (changed) {
            ops.erase(std::remove_if(ops.begin(), ops.end(),
                [](const std::unique_ptr<OpT>& o) { return !o; }), ops.end());
        }
        return changed;
    }
};
static PostConverterRegister<RemoveExtraConvertTensor> __rm_extra_convert("RemoveExtraConvertTensor");

// ── Registration ──
// Registers as "IspChainFusion" so optimizer finds it via optimizeNet()
static PostConverterRegister<IspChainFusion> __isp_fusion("IspChainFusion");

    // ══════════════════════════════════════════════════════════════════════
    //  NEW ISP BLOCKS - Missing optimization rules
    // ═════════════════════════════════════════════════════════════════════

    // DPC (Defective Pixel Correction) - median filter pattern
    