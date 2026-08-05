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
#include <map>
#include <set>
#include "../PostTreatUtils.hpp"
#include "../Global.hpp"
#include "MNN_generated.h"
#include "config.hpp"

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
        {"isp.lsc",              g_lsc_spv,              g_lsc_spv_len},
        // ISP opsets (10 new shaders — ae, awb, gamma, tone, dpc, denoise, eis_gyro, af_focus, calib_stats, ispc_stats)
        {"isp.ae",              g_isp_ae_spv,              g_isp_ae_spv_len},
        {"isp.gamma",           g_isp_gamma_spv,           g_isp_gamma_spv_len},
        {"isp.tone",            g_isp_tone_spv,            g_isp_tone_spv_len},
        {"isp.dpc",             g_isp_dpc_spv,             g_isp_dpc_spv_len},
        {"isp.denoise",         g_isp_denoise_spv,         g_isp_denoise_spv_len},
        {"isp.eis_gyro",        g_isp_eis_gyro_spv,        g_isp_eis_gyro_spv_len},
        {"isp.awb",             g_isp_awb_spv,             g_isp_awb_spv_len},
        {"isp.af_focus",        g_isp_af_focus_spv,        g_isp_af_focus_spv_len},
        {"isp.calib_stats",     g_isp_calib_stats_spv,     g_isp_calib_stats_spv_len},
        {"isp.ispc_stats",      g_isp_ispc_stats_spv,      g_isp_ispc_stats_spv_len},
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
// output_shape is NOT set here — ShapeExtra derives it from input tensor shape.
// Only global_size, group_size, and dispatch hints are embedded.
static void buildCommonAttrs(MNN::ExtraT* extra, int W, int H,
                              const std::vector<float>& uniforms) {
    addAttr(extra, "global_size", [&](MNN::AttributeT* a) {
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_INT32;
        a->tensor->int32s = {W, H, 1};
    });
    // Default: elementwise op — output shape copies the INPUT shape
    // (BLC, fcs, ee, gamma, vignetting, lsc, ...). Stride-2 ops
    // (unpack_blc / demosaic_ccm / pyramid) clear this marker and bake
    // their own global_size.
    addAttr(extra, "elementwise", [](MNN::AttributeT* a) { a->b = true; });
    addAttr(extra, "group_size", [&](MNN::AttributeT* a) {
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_INT32;
        a->tensor->int32s = {16, 16, 1};
    });
    addAttr(extra, "optimized_dispatch", [](MNN::AttributeT* a) { a->b = true; });
    // fp16_consts must stay FALSE — every embedded isp.* SPIR-V shader reads
    // its const block as raw FP32 (plain OpLoad, NO unpackHalf2x16). Packing
    // as FP16 corrupts W/H/gain → shader early-returns → all-zero output.
    addAttr(extra, "fp16_consts", [](MNN::AttributeT* a) { a->b = false; });
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

// Reduction-style Extra attrs (isp.calib_stats / isp.ispc_stats / isp.af_focus):
// The shaders are single-workgroup strided reductions: ONE 16x16 workgroup
// (256 threads) walks the whole image, then a shared-memory tree reduction
// produces the scalar result. Dispatch must be EXACTLY {16,16,1} (one
// workgroup) regardless of image size — never W×H workgroups.
// Also: only ONE input binding is declared (data tensor). The original
// Reduction op's second input (axes const tensor) must be DROPPED from
// inputIndexes by the caller, otherwise VulkanFuse writes it to binding 0
// (default for undeclared inputs), clobbering the const buffer.
// Reduction-style Extra attrs (isp.calib_stats / isp.ispc_stats / isp.af_focus):
// The shaders are single-workgroup strided reductions: ONE 16x16 workgroup
// (256 threads) walks the whole image, then a shared-memory tree reduction
// produces the result. Dispatch must be EXACTLY {16,16,1} (one workgroup)
// regardless of image size — never W×H workgroups.
// `reduce_keepdims` marker: ShapeExtra derives output [1,C,1,1] from the
// ACTUAL input tensor's channel dim at runtime (the reduce axis is spatial
// H×W with keepdims, so only the channel count matters). This survives
// dynamic ONNX inputs where C is concrete but H/W are -1 at convert time.
// Only ONE input binding is declared (data tensor). The original Reduction
// op's second input (axes const tensor) must be DROPPED from inputIndexes by
// the caller, otherwise VulkanFuse writes it to binding 0 (default for
// undeclared inputs), clobbering the const buffer.
static void buildReduceAttrs(MNN::ExtraT* extra,
                             const std::vector<float>& uniforms) {
    addAttr(extra, "global_size", [&](MNN::AttributeT* a) {
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_INT32;
        a->tensor->int32s = {16, 16, 1};  // single 16x16 workgroup
    });
    addAttr(extra, "group_size", [&](MNN::AttributeT* a) {
        a->tensor.reset(new MNN::BlobT);
        a->tensor->dataType = MNN::DataType_DT_INT32;
        a->tensor->int32s = {16, 16, 1};
    });
    addAttr(extra, "reduce_keepdims", [](MNN::AttributeT* a) { a->b = true; });
    addAttr(extra, "optimized_dispatch", [](MNN::AttributeT* a) { a->b = true; });
    // fp16_consts must stay FALSE (shaders read raw FP32, no unpackHalf2x16)
    addAttr(extra, "fp16_consts", [](MNN::AttributeT* a) { a->b = false; });
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

// Clear the elementwise output-shape marker (stride-2 ops like
// unpack_blc / demosaic_ccm / pyramid / unpack_demosaic halve resolution).
static void clearElementwise(MNN::ExtraT* extra) {
    if (extra->attr.empty()) return;
    for (auto& a : extra->attr) {
        if (a->key == "elementwise") { a->b = false; return; }
    }
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

// ── Exact graph-connectivity helpers (replace array-position heuristics) ──
// Producer: the op whose outputIndexes contains tensorId.
static OpT* producerOf(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId) {
    for (auto& op : ops) {
        if (!op) continue;
        for (int out : op->outputIndexes) {
            if (out == tensorId) return op.get();
        }
    }
    return nullptr;
}

// Const producer: find a Const op that outputs tensorId (returns its blob data).
static const BlobT* constBlobOf(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId) {
    for (auto& op : ops) {
        if (!op || op->type != MNN::OpType_Const) continue;
        for (int out : op->outputIndexes) {
            if (out == tensorId) {
                auto* b = op->main.AsBlob();
                if (b) return b;
            }
        }
    }
    return nullptr;
}

// Direct producer check: tensorId's producer op must be exactly `type`.
static bool producerIs(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId, MNN::OpType type) {
    auto* p = producerOf(ops, tensorId);
    return p != nullptr && p->type == type;
}

// Single-input producer: returns input tensor id of op `cur` at position `inputSlot`.
// Null-safe for ops with missing inputs.
static int inputTensorOf(const OpT* cur, int slot) {
    if (!cur || slot < 0 || slot >= (int)cur->inputIndexes.size()) return -1;
    return cur->inputIndexes[slot];
}

// Check op at index idx has exactly the given type (null-safe).
static bool opIs(const std::vector<std::unique_ptr<OpT>>& ops, int idx, MNN::OpType type) {
    return idx >= 0 && idx < (int)ops.size() && ops[idx] && ops[idx]->type == type;
}

// Find op index whose outputIndexes contains tensorId (returns -1 if none).
static int opIndexProducerOf(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId) {
    for (int j = 0; j < (int)ops.size(); j++) {
        if (!ops[j]) continue;
        for (int out : ops[j]->outputIndexes) {
            if (out == tensorId) return j;
        }
    }
    return -1;
}

// BinaryOp type check helper.
static bool isBinOp(const OpT* op, MNN::BinaryOpOperation t) {
    if (!op || op->type != MNN::OpType_BinaryOp) return false;
    if (op->main.type != MNN::OpParameter_BinaryOp) return false;
    return op->main.AsBinaryOp()->opType == t;
}

// UnaryOp type check helper.
static bool isUnOp(const OpT* op, MNN::UnaryOpOperation t) {
    if (!op || op->type != MNN::OpType_UnaryOp) return false;
    if (op->main.type != MNN::OpParameter_UnaryOp) return false;
    return op->main.AsUnaryOp()->opType == t;
}

// Find the FIRST consumer of tensorId (any op whose inputIndexes contains it).
static int consumerOf(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId) {
    for (int k = 0; k < (int)ops.size(); k++) {
        if (!ops[k]) continue;
        for (int inIdx : ops[k]->inputIndexes) {
            if (inIdx == tensorId) return k;
        }
    }
    return -1;
}

// Find consumer of tensorId that is a BinaryOp with specific type.
static int consumerOfBinOp(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId, MNN::BinaryOpOperation binOp) {
    for (int k = 0; k < (int)ops.size(); k++) {
        if (!ops[k] || ops[k]->type != MNN::OpType_BinaryOp) continue;
        if (!isBinOp(ops[k].get(), binOp)) continue;
        for (int inIdx : ops[k]->inputIndexes) {
            if (inIdx == tensorId) return k;
        }
    }
    return -1;
}

// Find consumer of tensorId that is a specific OpType.
static int consumerOfType(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId, MNN::OpType type) {
    for (int k = 0; k < (int)ops.size(); k++) {
        if (!ops[k] || ops[k]->type != type) continue;
        for (int inIdx : ops[k]->inputIndexes) {
            if (inIdx == tensorId) return k;
        }
    }
    return -1;
}

// Check if tensorId is consumed by any op outside the fused set.
static bool hasExternalConsumer(const std::vector<std::unique_ptr<OpT>>& ops,
                                int tensorId,
                                const std::vector<int>& fusedOps) {
    for (int k = 0; k < (int)ops.size(); k++) {
        if (!ops[k]) continue;
        bool isFused = false;
        for (int f : fusedOps) { if (f == k) { isFused = true; break; } }
        if (isFused) continue;
        for (int inIdx : ops[k]->inputIndexes) {
            if (inIdx == tensorId) return true;
        }
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════
//  Pass 1: Standard MNN ops → ISP Extra ops
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
//  Exact-pattern matching framework
// ═══════════════════════════════════════════════════════════════════
// Each try* helper first checks a table of ExactPattern structs
// (sorted by opTypes.size() DESCENDING — longest match first).
// Only if no exact pattern matches does it fall through to the old
// heuristic code.
struct ExactPattern {
    std::vector<MNN::OpType> opTypes;
    int constElems;
    int constIndex;
    const char* ispType;
    const char* spvName;
    const char* namedKey;
    int convWeightElems = -1;
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, const char* nk = nullptr, int cwe = -1)
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), convWeightElems(cwe) {}
};

// Navigate ops starting at , skipping Const and ConvertTensor,
// collecting up to  op types. Returns actual index per collected op.
static bool collectChain(const std::vector<std::unique_ptr<OpT>>& ops,
                         int start, int count,
                         std::vector<int>& indices, std::vector<MNN::OpType>& types) {
    int j = start;
    for (int n = 0; n < count && j < (int)ops.size(); j++) {
        if (!ops[j]) continue;
        if (ops[j]->type == MNN::OpType_Const ||
            ops[j]->type == MNN::OpType_ConvertTensor) continue;
        indices.push_back(j);
        types.push_back(ops[j]->type);
        n++;
    }
    return (int)types.size() == count;
}

// Check if an exact pattern matches at ops[i].
static bool matchExact(const std::vector<std::unique_ptr<OpT>>& ops,
                       int i, const ExactPattern& pat) {
    if (i < 0 || i >= (int)ops.size() || !ops[i]) return false;
    std::vector<int> idx;
    std::vector<MNN::OpType> types;
    if (!collectChain(ops, i, (int)pat.opTypes.size(), idx, types)) return false;
    for (int k = 0; k < (int)pat.opTypes.size(); k++) {
        if (types[k] != pat.opTypes[k]) return false;
    }
    // Check const constraint: inputIndexes[constIndex] must be a Const with constElems floats
    if (pat.constIndex >= 0 && pat.constElems >= 0) {
        if (idx.empty()) return false;
        auto* op = ops[idx[0]].get();
        if (pat.constIndex >= (int)op->inputIndexes.size()) return false;
        int tensorId = op->inputIndexes[pat.constIndex];
        auto* blb = constBlobOf(ops, tensorId);
        if (!blb || (int)blb->float32s.size() != pat.constElems) return false;
    }
    // Check conv weight count
    if (pat.convWeightElems >= 0) {
        if (idx.empty()) return false;
        auto* op = ops[idx[0]].get();
        if (op->type != MNN::OpType_Convolution) return false;
        auto* c = op->main.AsConvolution2D();
        if (!c || (int)c->weight.size() != pat.convWeightElems) return false;
    }
    return true;
}

// Convert exact match to isp.* Extra op, consuming chain ops.
static bool applyExact(std::vector<std::unique_ptr<OpT>>& ops, int& i,
                       const ExactPattern& pat, int mW, int mH,
                       const std::vector<float>& u) {
    std::vector<int> idx;
    std::vector<MNN::OpType> types;
    if (!collectChain(ops, i, (int)pat.opTypes.size(), idx, types)) return false;
    ops[idx[0]]->type = MNN::OpType_Extra;
    ops[idx[0]]->main.type = MNN::OpParameter_Extra;
    auto* ex = new MNN::ExtraT();
    ex->type = pat.ispType;
    ex->engine = "MNN";
    buildCommonAttrs(ex, mW, mH, u);
    setEngine(ex);
    addSpirv(ex, pat.spvName);
    if (pat.namedKey && pat.namedKey[0]) {
        std::vector<float> nf(u.begin() + 2, u.end());
        addNamedFloats(ex, pat.namedKey, nf);
    }
    ops[idx[0]]->main.value = ex;
    // Output: last op in chain
    if ((int)idx.size() > 1) {
        ops[idx[0]]->outputIndexes[0] = ops[idx.back()]->outputIndexes[0];
    }
    // Nullify consumed ops (except first)
    for (int k = 1; k < (int)idx.size(); k++) ops[idx[k]].reset();
    i = idx.back();
    VLOG(2) << "[P1] EXACT " << pat.ispType << " at " << idx[0]
            << " chain=" << idx.size();
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// Global exact-pattern tables (checked first, before any try* heuristic)
// Sorted by chain length DESCENDING within each table.
static const ExactPattern kExactFcs[] = {
    // 1-op: Conv(1x1) with 9 weights (CCM, CcmBlock)
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.fcs", "isp.fcs", "fcs", 9),
    // 1-op: Mul with 3-elem const gains (WbGainsBlock)
    ExactPattern({MNN::OpType_BinaryOp},
                 3, 1, "isp.fcs", "isp.fcs", "fcs"),
    // 1-op: Mul with 4-elem const gains (BayerWbBlock)
    ExactPattern({MNN::OpType_BinaryOp},
                 4, 1, "isp.fcs", "isp.fcs", "fcs"),
};

static const ExactPattern kExactDemosaic[] = {
    // Conv(depthwise, 3x3, 4ch) with 36 weights (real debayer)
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 36),
    // Conv(5x5) with 300 weights, 4→3 (DebayerBlock learned debayer)
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 300),
};

static const ExactPattern kExactDisplay[] = {
    // Permute→Padding→Gather→BinaryOp→Cast (DisplayBlock: Identity→Transp→Pad→Gather→Mul→Cast)
    ExactPattern({MNN::OpType_Permute, MNN::OpType_Padding,
                  MNN::OpType_Gather, MNN::OpType_BinaryOp, MNN::OpType_Cast},
                 -1, -1, "isp.display", "isp.display", "display"),
    // Transpose→Pad→Gather→Mul→Cast (OnnxDisplayBlock)
    ExactPattern({MNN::OpType_Permute, MNN::OpType_Padding, MNN::OpType_Gather,
                  MNN::OpType_BinaryOp, MNN::OpType_Cast},
                 -1, -1, "isp.display", "isp.display", "display"),
    // Transpose→Pad→Mul→Cast (OnnxDisplayBlock without Gather)
    ExactPattern({MNN::OpType_Permute, MNN::OpType_Padding,
                  MNN::OpType_BinaryOp, MNN::OpType_Cast},
                 -1, -1, "isp.display", "isp.display", "display"),
    // Conv(1x1,3→3)→Permute→Padding→Gather→Mul→Cast (DisplayBlock isYuv=true)
    ExactPattern({MNN::OpType_Convolution, MNN::OpType_Permute,
                  MNN::OpType_Padding, MNN::OpType_Gather, MNN::OpType_BinaryOp, MNN::OpType_Cast},
                 -1, -1, "isp.display", "isp.display", "display"),
};

static const ExactPattern kExactUnpack[] = {
    ExactPattern({MNN::OpType_Reshape, MNN::OpType_ReLU6, MNN::OpType_BinaryOp},
                 -1, -1, "isp.unpack_blc", "isp.unpack_blc", nullptr),
    ExactPattern({MNN::OpType_BinaryOp},
                 1, 1, "isp.unpack_blc", "isp.unpack_blc", nullptr),
};

static const ExactPattern kExactPyramid[] = {
    ExactPattern({MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
                  MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
                 -1, -1, "isp.pyramid", "isp.pyramid", "fcs"),
};

static const ExactPattern kExactAe[] = {
    ExactPattern({MNN::OpType_BinaryOp},
                 1, 1, "isp.ae", "isp.ae", "ae"),
};

static const ExactPattern kExactAfFocus[] = {
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.af_focus", "isp.af_focus", nullptr, 9),
};

static bool tryExactFirst(std::vector<std::unique_ptr<MNN::OpT>>& ops,
                          int& i, int mW, int mH) {
#define TRY_EXACT_TABLE(tbl) do { \
        for (auto& pat : tbl) { \
            if (matchExact(ops, i, pat)) { \
                float str = 1.0f; \
                std::vector<int> idx; std::vector<MNN::OpType> types; \
                collectChain(ops, i, (int)pat.opTypes.size(), idx, types); \
                if (!idx.empty()) { \
                    auto* op = ops[idx[0]].get(); \
                    if (op->type == MNN::OpType_BinaryOp) { \
                        for (int inIdx : op->inputIndexes) { \
                            auto* blb = constBlobOf(ops, inIdx); \
                            if (blb && blb->float32s.size() >= 1) { str = blb->float32s[0]; break; } \
                        } \
                    } else if (op->type == MNN::OpType_Convolution) { \
                        auto* c = op->main.AsConvolution2D(); \
                        if (c && !c->weight.empty()) str = c->weight[0]; \
                    } \
                } \
                std::vector<float> u = {float(mW), float(mH), str, 0, 0,0,0,0}; \
                return applyExact(ops, i, pat, mW, mH, u); \
            } \
        } \
    } while(0)

    TRY_EXACT_TABLE(kExactFcs);
    TRY_EXACT_TABLE(kExactDisplay);       // before demosaic to avoid Conv conflict
    TRY_EXACT_TABLE(kExactDemosaic);      // specific weight counts, after display
    TRY_EXACT_TABLE(kExactUnpack);        // 1-op Sub(1-elem) for RawBlcBlock
    TRY_EXACT_TABLE(kExactPyramid);
    TRY_EXACT_TABLE(kExactAe);
    TRY_EXACT_TABLE(kExactAfFocus);
    return false;
}
#undef TRY_EXACT_TABLE

// ═══════════════════════════════════════════════════════════════════
class Pass1_ToExtra : public PostConverter {
public:
    static Pass1_ToExtra* instance() {
        static Pass1_ToExtra p;
        return &p;
    }

    // Pattern level filter: "basic" = pre-existing ISP patterns only,
    // "full" = all patterns including new reduction/scalar ops.
    // Set via ONNX metadata: isp_fusion_level=basic|full
    mutable int mPatternThreshold = 999;  // run try* with id <= threshold

    // Try a numbered pattern: runs only if id <= threshold.
    // threshold=0 means no patterns, 999 means all.
    typedef std::function<bool(std::vector<std::unique_ptr<MNN::OpT>>&, int&)> TryFn;
    bool tryN(int id, TryFn fn, std::vector<std::unique_ptr<MNN::OpT>>& ops, int& i) const {
        if (id <= mPatternThreshold) return fn(ops, i);
        return false;
    }
    // Helper: wrap member function as TryFn
    typedef bool (Pass1_ToExtra::*MemberFn)(std::vector<std::unique_ptr<MNN::OpT>>&, int&) const;
    bool tryN(int id, MemberFn fn, std::vector<std::unique_ptr<MNN::OpT>>& ops, int& i) const {
        if (id <= mPatternThreshold) return (this->*fn)(ops, i);
        return false;
    }

    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;

        // Read pattern threshold from modelConfig
        {
            auto* cfg = Global<modelConfig>::Get();
            if (cfg != nullptr) {
                mPatternThreshold = cfg->ispFusionThreshold;
                VLOG(2) << "[P1] Pattern threshold from modelConfig: " << mPatternThreshold;
            }
        }

        // Extract input dimensions from the Input op
        // IMPORTANT: ONNX dynamic dims are -1 (e.g. [1,1,-1,-1]); only accept
        // concrete positive dims. If dynamic, leave defaults (3840x2160) so
        // baked global_size/output_shape stay positive and the Vulkan runtime
        // re-derives exact dims from the actual input tensor at dispatch time.
        for (auto& op : ops) {
            if (op && op->type == MNN::OpType_Input) {
                auto* inp = op->main.AsInput();
                if (inp && inp->dims.size() >= 4) {
                    if (inp->dims[2] > 0 && inp->dims[3] > 0) {
                        mInH = inp->dims[2];  // NCHW
                        mInW = inp->dims[3];
                    } else {
                        VLOG(2) << "[P1] Dynamic input dims (" << inp->dims[2] << "x" << inp->dims[3]
                                << "), keeping default " << mInH << "x" << mInW;
                    }
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
            // Ordering principle: longer scan / more ops consumed → earlier.
            // Same entry type: longer scan window first.
            // Special: tryClipAbsorbFwd before block rules (absorbs into Conv before
            //          block rules convert Conv to Extra).
            // Special: tryTone before tryGamma (both POW; tone is multi-op chain).
            // Special: tryIspControllerStats before tryCalibStats (more specific).

            // ═══════════════════════════════════════════════════════════════
            // NUMBERED PATTERNS — incremental threshold testing.
            // Patterns 1-41: pre-existing (basic). Patterns 42-50: new ISP ops.
            // Threshold=0: none. Threshold=41: basic only. Threshold=999: all.
            // ═══════════════════════════════════════════════════════════════

            // ── Exact-pattern pre-dispatch (before any heuristic) ──
            if (tryExactFirst(ops, i, mW, mH)) { changed = true; continue; }

            // ── GROUP 1: Unpack block patterns (~8 ops → 2 ops) ──
            if (tryN(1, &Pass1_ToExtra::tryUnpackPackedChain, ops, i)) { changed = true; continue; }
            if (tryN(2, &Pass1_ToExtra::tryUnpack, ops, i)) { changed = true; continue; }
            if (tryN(3, &Pass1_ToExtra::tryUnpackRust, ops, i)) { changed = true; continue; }

            // ── GROUP 2: Multi-op block patterns (3-4 ops) ──
            if (tryN(4, &Pass1_ToExtra::tryLdci, ops, i)) { changed = true; continue; }
            if (tryN(5, &Pass1_ToExtra::tryRustReduceLdci, ops, i)) { changed = true; continue; }
            if (tryN(6, &Pass1_ToExtra::tryAutoContrast, ops, i)) { changed = true; continue; }
            if (tryN(7, &Pass1_ToExtra::trySubMaxMin, ops, i)) { changed = true; continue; }

            // ── GROUP 3: Absorption rules (MUST run before block rules) ──
            if (tryN(8, &Pass1_ToExtra::tryClipAbsorbFwd, ops, i)) { changed = true; continue; }

            // ── GROUP 4: Block patterns (1-2 ops, scan i+4) ──
            if (tryN(9, &Pass1_ToExtra::tryDemosaic, ops, i)) { changed = true; continue; }
            if (tryN(10, &Pass1_ToExtra::tryDemosaicInterp, ops, i)) { changed = true; continue; }
            if (tryN(11, &Pass1_ToExtra::tryEe, ops, i)) { changed = true; continue; }
            if (tryN(12, &Pass1_ToExtra::tryRustConvEe, ops, i)) { changed = true; continue; }
            if (tryN(13, &Pass1_ToExtra::tryRustConvFcs, ops, i)) { changed = true; continue; }
            if (tryN(14, &Pass1_ToExtra::tryGaussianDenoise, ops, i)) { changed = true; continue; }
            if (tryN(15, &Pass1_ToExtra::tryRustExtraEe, ops, i)) { changed = true; continue; }
            if (tryN(16, &Pass1_ToExtra::trySubClipNormalize, ops, i)) { changed = true; continue; }
            if (tryN(17, &Pass1_ToExtra::trySubMul, ops, i)) { changed = true; continue; }
            if (tryN(18, &Pass1_ToExtra::tryMulAdd, ops, i)) { changed = true; continue; }
            if (tryN(19, &Pass1_ToExtra::tryMulClip, ops, i)) { changed = true; continue; }
            if (tryN(20, &Pass1_ToExtra::tryConvSub, ops, i)) { changed = true; continue; }

            // ── GROUP 5: Single-op patterns (1 op) ──
            if (tryN(21, &Pass1_ToExtra::tryFcs, ops, i)) { changed = true; continue; }
            if (tryN(22, &Pass1_ToExtra::tryArgbConvert, ops, i)) { changed = true; continue; }
            if (tryN(23, &Pass1_ToExtra::tryYuv420Convert, ops, i)) { changed = true; continue; }
            if (tryN(24, &Pass1_ToExtra::tryGrayscale, ops, i)) { changed = true; continue; }
            if (tryN(25, &Pass1_ToExtra::tryDisplay, ops, i)) { changed = true; continue; }
            if (tryN(26, &Pass1_ToExtra::tryPyramid, ops, i)) { changed = true; continue; }
            if (tryN(27, &Pass1_ToExtra::tryWarp, ops, i)) { changed = true; continue; }
            if (tryN(28, &Pass1_ToExtra::tryRustDisplay, ops, i)) { changed = true; continue; }
            if (tryN(29, &Pass1_ToExtra::tryVignetting, ops, i)) { changed = true; continue; }
            if (tryN(30, &Pass1_ToExtra::tryLsc, ops, i)) { changed = true; continue; }

            // ── GROUP 6: cam_app ISP block patterns ──
            if (tryN(31, &Pass1_ToExtra::tryUnsharp, ops, i)) { changed = true; continue; }
            if (tryN(32, &Pass1_ToExtra::trySaturation, ops, i)) { changed = true; continue; }
            if (tryN(33, &Pass1_ToExtra::tryBadPixel, ops, i)) { changed = true; continue; }
            if (tryN(34, &Pass1_ToExtra::tryBayerWb, ops, i)) { changed = true; continue; }
            if (tryN(35, &Pass1_ToExtra::tryBilateral, ops, i)) { changed = true; continue; }
            if (tryN(36, &Pass1_ToExtra::tryYuvSat, ops, i)) { changed = true; continue; }
            if (tryN(37, &Pass1_ToExtra::tryYuvSat7, ops, i)) { changed = true; continue; }
            if (tryN(38, &Pass1_ToExtra::tryBayerWbReshape, ops, i)) { changed = true; continue; }
            if (tryN(39, &Pass1_ToExtra::tryBLC, ops, i)) { changed = true; continue; }
            if (tryN(40, &Pass1_ToExtra::tryNormalize, ops, i)) { changed = true; continue; }
            if (tryN(41, &Pass1_ToExtra::tryDemosaicStandalone, ops, i)) { changed = true; continue; }

            // ── GROUP 7: NEW ISP ops (require SPIR-V shaders) ──
            if (tryN(42, &Pass1_ToExtra::tryAfFocus, ops, i)) { changed = true; continue; }
            if (tryN(43, &Pass1_ToExtra::tryDpc, ops, i)) { changed = true; continue; }
            if (tryN(44, &Pass1_ToExtra::tryAwb, ops, i)) { changed = true; continue; }
            if (tryN(45, &Pass1_ToExtra::tryTone, ops, i)) { changed = true; continue; }
            if (tryN(46, &Pass1_ToExtra::tryAe, ops, i)) { changed = true; continue; }
            if (tryN(47, &Pass1_ToExtra::tryGamma, ops, i)) { changed = true; continue; }
            if (tryN(48, &Pass1_ToExtra::tryIspControllerStats, ops, i)) { changed = true; continue; }
            if (tryN(49, &Pass1_ToExtra::tryCalibStats, ops, i)) { changed = true; continue; }
            if (tryN(50, &Pass1_ToExtra::tryEisGyro, ops, i)) { changed = true; continue; }

        }

        // ══════ FUSION SUMMARY ══════
        // Print every op's type and which ISP opset fused it (if any).
        {
            std::map<std::string, int> ispCounts;
            int primitiveCount = 0;
            int totalOps = 0;
            fprintf(stderr, "[P1] ═══ FUSION SUMMARY (threshold=%d) ═══\n", mPatternThreshold);
            for (int idx = 0; idx < (int)ops.size(); idx++) {
                if (!ops[idx]) continue;
                totalOps++;
                auto t = ops[idx]->type;
                if (t == MNN::OpType_Extra) {
                    auto* ex = ops[idx]->main.AsExtra();
                    std::string name = ex ? ex->type : "unknown";
                    ispCounts[name]++;
                    fprintf(stderr, "  [%3d] EXTRA  %-30s  in=%zu out=%zu\n",
                            idx, name.c_str(),
                            ops[idx]->inputIndexes.size(),
                            ops[idx]->outputIndexes.size());
                } else {
                    primitiveCount++;
                    fprintf(stderr, "  [%3d] %-30s  in=%zu out=%zu\n",
                            idx, MNN::EnumNameOpType(t),
                            ops[idx]->inputIndexes.size(),
                            ops[idx]->outputIndexes.size());
                }
            }
            fprintf(stderr, "[P1] ── ISP opset counts ──\n");
            for (auto& kv : ispCounts) {
                fprintf(stderr, "  %-30s x%d\n", kv.first.c_str(), kv.second);
            }
            fprintf(stderr, "[P1] Total: %d ops (%d ISP Extra + %d primitive)\n",
                    totalOps, totalOps - primitiveCount, primitiveCount);
            fflush(stderr);  // ensure fusion summary is flushed before converter returns
        }

        // ══════ POST-FUSION GRAPH VALIDATION ══════
        // Catch dangling tensor references from any pattern that nulled an op
        // whose output is still consumed by a live op.
        {
            // Collect all tensor IDs produced by live ops
            std::set<int> produced;
            for (auto& op : ops) {
                if (!op) continue;
                for (int out : op->outputIndexes) produced.insert(out);
            }
            // Check every live op's inputs
            int danglingCount = 0;
            for (int i = 0; i < (int)ops.size(); i++) {
                if (!ops[i]) continue;
                for (int inIdx : ops[i]->inputIndexes) {
                    if (produced.find(inIdx) == produced.end()) {
                        // inIdx not produced by any live op — could be model input (small ID) or dangling
                        if (inIdx > (int)produced.size() + 10) {
                            // Likely dangling — model inputs are usually small IDs
                            VLOG(1) << "[P1] GRAPH_INTEGRITY: op[" << i << "] type=" << ops[i]->type
                                    << " references dangling tensor " << inIdx;
                            danglingCount++;
                        }
                    }
                }
            }
            if (danglingCount > 0) {
                VLOG(1) << "[P1] GRAPH_INTEGRITY: " << danglingCount << " dangling tensor references found";
            }
        }

        ops.erase(std::remove_if(ops.begin(), ops.end(),
                  [](const std::unique_ptr<OpT>& o) { return !o; }), ops.end());

        // ── Const-input trim for fused Extra ops ──
        // Every isp.* SPIR-V shader binds exactly ONE data input (binding 1),
        // with all constants embedded in the const attr (binding 0). Patterns
        // that convert BinaryOps (Mul/Sub/Div) in place keep the original
        // inputIndexes=[data, const, ...]; the extra const tensors get the
        // default binding 0 in VulkanFuse, CLOBBERING the uniforms buffer
        // (shader then reads garbage W/H → 0 workgroups or empty writes).
        // Drop every input whose tensor is produced by an OpType_Const op.
        {
            int trimmed = 0;
            std::set<int> constProducers;
            for (auto& op : ops) {
                if (op && op->type == MNN::OpType_Const) {
                    for (int outIdx : op->outputIndexes) constProducers.insert(outIdx);
                }
            }
            for (auto& op : ops) {
                if (!op || op->type != MNN::OpType_Extra) continue;
                auto* ex = op->main.AsExtra();
                if (!ex) continue;
                // Only trim ops whose SPIR-V uses the single-data-input convention.
                std::vector<int> kept;
                for (int inIdx : op->inputIndexes) {
                    if (constProducers.count(inIdx)) { trimmed++; continue; }
                    kept.push_back(inIdx);
                }
                if (kept.size() != op->inputIndexes.size()) {
                    op->inputIndexes = kept;
                }
            }
            if (trimmed > 0) {
                VLOG(1) << "[P1] Const-input trim: removed " << trimmed
                        << " const tensor refs from fused Extra ops";
            }
        }
        return changed;
    }

private:
    // Dimensions inferred from model input shape at onExecute time
    mutable int mW = 1920, mH = 1080;       // output (after stride-2)
    mutable int mInW = 3840, mInH = 2160;   // input (Bayer raw)

    // R1: Cast [+Div(Normalize)] + Conv(2×2,stride=2,3-4ch) → isp.unpack_blc or isp.demosaic(binning)
    // Enhanced: also absorps Normalize (Div÷max) before Cast, and BLC (Sub+Clip) after Conv.
    bool tryUnpack(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        // EXACT PATTERN (connectivity-based, no array-position heuristics):
        //   Input → [Cast] → [Div(Normalize, /sensor_max)] → Conv(unpack 2×2 s2 4ch)
        //   Conv → Sub(BLC const) → [ReLU6]
        // Required: the Conv must be an unpack conv AND the chain must have
        // either a Cast/Div upstream anchor OR a BLC Sub downstream anchor.
        // If neither anchor exists, this is a regular Conv — return false.
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        auto* conv = ops[i]->main.AsConvolution2D();
        bool is3chOut = false;
        if (!isUnpackConv(conv, &is3chOut)) return false;

        int ci = i;
        int convIn0 = inputTensorOf(ops[ci].get(), 0);

        // ── Downstream anchors: Sub(BLC const) then optional ReLU6/Clip ──
        float blc0 = 0.0f, blc1 = 0.0f, blc2 = 0.0f, blc3 = 0.0f;
        int blcSubIdx = -1, blcClipIdx = -1;
        {
            int convOut = ops[ci]->outputIndexes.empty() ? -1 : ops[ci]->outputIndexes[0];
            int subIdx = opIndexProducerOf(ops, convOut);  // direct consumer
            if (subIdx >= 0 && isBinOp(ops[subIdx].get(), MNN::BinaryOpOperation_SUB)) {
                // second input must be a Const with >=4 floats
                int subIn1 = inputTensorOf(ops[subIdx].get(), 1);
                auto* blb = constBlobOf(ops, subIn1);
                if (blb && blb->float32s.size() >= 4) {
                    blc0 = blb->float32s[0];
                    blc1 = blb->float32s[1];
                    blc2 = blb->float32s[2];
                    blc3 = blb->float32s[3];
                    blcSubIdx = subIdx;
                    // optional ReLU6/Clip after Sub
                    int subOut = ops[subIdx]->outputIndexes.empty() ? -1 : ops[subIdx]->outputIndexes[0];
                    int clipIdx = opIndexProducerOf(ops, subOut);
                    if (clipIdx >= 0 && (ops[clipIdx]->type == MNN::OpType_ReLU ||
                                         ops[clipIdx]->type == MNN::OpType_ReLU6)) {
                        blcClipIdx = clipIdx;
                    }
                }
            }
        }

        // ── Upstream anchors: Cast then optional Div(Normalize, /sensor_max) ──
        float normScale = 1023.0f;  // default sensor_max
        int castIdx = -1, divIdx = -1;
        {
            // The Conv's input chain: find producer of convIn0.
            int cur = convIn0;
            // Walk backwards through Cast → Div → original input.
            // Layout: Input → Div → Cast → Conv  OR  Input → Cast → Conv.
            auto* castOp = producerOf(ops, cur);
            if (castOp && castOp->type == MNN::OpType_Cast) {
                castIdx = opIndexProducerOf(ops, cur);
                int castIn = inputTensorOf(castOp, 0);
                auto* divOp = producerOf(ops, castIn);
                if (divOp && isBinOp(divOp, MNN::BinaryOpOperation_DIV)) {
                    divIdx = opIndexProducerOf(ops, castIn);
                    // second input must be Const with positive float
                    int divIn1 = inputTensorOf(divOp, 1);
                    auto* blb = constBlobOf(ops, divIn1);
                    if (blb && !blb->float32s.empty() && blb->float32s[0] > 0.0f) {
                        normScale = blb->float32s[0];
                    }
                }
            }
        }

        // Must find at least one anchor (upstream Cast/Div OR downstream BLC Sub).
        if (castIdx < 0 && blcSubIdx < 0) return false;

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

        // Reroute Extra input to skip consumed ops: trace back through
        // Cast → Div → original input.
        if (castIdx >= 0) {
            int traceIn = convIn0;
            if (divIdx >= 0 && ops[divIdx] && !ops[divIdx]->inputIndexes.empty()) {
                traceIn = ops[divIdx]->inputIndexes[0];
            }
            if (castIdx >= 0 && ops[castIdx] && !ops[castIdx]->inputIndexes.empty()) {
                traceIn = ops[castIdx]->inputIndexes[0];
            }
            if (traceIn >= 0 && !ops[ci]->inputIndexes.empty()) {
                VLOG(2) << "[P1] R1: reroute Extra input " << ops[ci]->inputIndexes[0]
                        << " -> " << traceIn;
                ops[ci]->inputIndexes[0] = traceIn;
            }
        }

        // Reroute downstream consumers of dead tensors to the Extra output.
        int convOutTensor = ops[ci]->outputIndexes[0];
        std::vector<int> deadTensors;
        if (blcSubIdx >= 0) deadTensors.push_back(ops[blcSubIdx]->outputIndexes[0]);
        if (blcClipIdx >= 0) deadTensors.push_back(ops[blcClipIdx]->outputIndexes[0]);
        if (divIdx >= 0) deadTensors.push_back(ops[divIdx]->outputIndexes[0]);
        if (castIdx >= 0) {
            for (auto outIdx : ops[castIdx]->outputIndexes) deadTensors.push_back(outIdx);
        }
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

        // EXACT MATCH: walk the producer chain of the Conv's input tensor.
        // Expected: Div(norm) -> Sub(BLC) -> ReLU6/ReLU -> Conv(CCM).
        // Every hop must be verified via tensor connectivity, not array position.
        float blc0 = 0.0f, blc1 = 0.0f, blc2 = 0.0f, blc3 = 0.0f;
        float normScale = 1023.0f;
        int blcSubIdx = -1, blcClipIdx = -1, normDivIdx = -1;

        // Hop 1: Conv input <- Clip/ReLU
        int convIn0 = inputTensorOf(ops[i].get(), 0);
        int clipIdx = opIndexProducerOf(ops, convIn0);
        if (clipIdx >= 0 && ops[clipIdx] &&
            (ops[clipIdx]->type == MNN::OpType_ReLU ||
             ops[clipIdx]->type == MNN::OpType_ReLU6)) {
            blcClipIdx = clipIdx;
            // Hop 2: Clip input <- Sub(BLC)
            int clipIn0 = inputTensorOf(ops[clipIdx].get(), 0);
            int subIdx = opIndexProducerOf(ops, clipIn0);
            if (subIdx >= 0 && ops[subIdx] &&
                ops[subIdx]->type == MNN::OpType_BinaryOp &&
                isBinOp(ops[subIdx].get(), MNN::BinaryOpOperation_SUB)) {
                // Extract blc from Sub's second (Const) input
                int subIn1 = inputTensorOf(ops[subIdx].get(), 1);
                auto* blb = constBlobOf(ops, subIn1);
                if (blb && blb->float32s.size() >= 4) {
                    blc0 = blb->float32s[0];
                    blc1 = blb->float32s[1];
                    blc2 = blb->float32s[2];
                    blc3 = blb->float32s[3];
                    blcSubIdx = subIdx;
                }
            }
            // Hop 3: Sub/Clip input <- Div(normalize /sensor_max)
            int anchorIdx = (blcSubIdx >= 0) ? blcSubIdx : blcClipIdx;
            int anchorIn0 = inputTensorOf(ops[anchorIdx].get(), 0);
            int divIdx = opIndexProducerOf(ops, anchorIn0);
            if (divIdx >= 0 && ops[divIdx] &&
                ops[divIdx]->type == MNN::OpType_BinaryOp &&
                isBinOp(ops[divIdx].get(), MNN::BinaryOpOperation_DIV)) {
                int divIn1 = inputTensorOf(ops[divIdx].get(), 1);
                auto* dblb = constBlobOf(ops, divIn1);
                if (dblb && !dblb->float32s.empty() && dblb->float32s[0] > 0) {
                    normScale = dblb->float32s[0];
                    normDivIdx = divIdx;
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
        // ── Exact patterns (longest chain first) ──
        static const ExactPattern kExact[] = {
            // 4-op: Sub→Mul→Add→Clip saturation chain
            ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
                         -1, -1, "isp.fcs", "isp.fcs", "fcs"),
            // 3-op: Sub→Mul→Clip (UnsharpBlock-like without Add)
            ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
                         -1, -1, "isp.fcs", "isp.fcs", "fcs"),
            // 2-op: Sub→Mul (UnsharpBlock)
            ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_BinaryOp},
                         -1, -1, "isp.fcs", "isp.fcs", "fcs"),
            // 1-op: Conv(1x1) with 9 weights (CCM, CcmBlock)
            ExactPattern({MNN::OpType_Convolution},
                         -1, -1, "isp.fcs", "isp.fcs", "fcs", 9),
            // 1-op: Conv(1x1) with 27 weights (3x3 CCM)
            ExactPattern({MNN::OpType_Convolution},
                         -1, -1, "isp.fcs", "isp.fcs", "fcs", 27),
            // 1-op: Mul with 3-elem const gains (BayerWbBlock)
            ExactPattern({MNN::OpType_BinaryOp},
                         3, 1, "isp.fcs", "isp.fcs", "fcs"),
            // 1-op: Sub with 1-elem const (BLC normalize)
            ExactPattern({MNN::OpType_BinaryOp},
                         1, 1, "isp.fcs", "isp.fcs", "fcs"),
        };
        // Try exact patterns first (table is already sorted by chain length)
        for (auto& pat : kExact) {
            if (!matchExact(ops, i, pat)) continue;
            // Extract gains/strength for the uniform from the first op's const input
            float str = 1.0f;
            std::vector<int> idx; std::vector<MNN::OpType> types;
            collectChain(ops, i, (int)pat.opTypes.size(), idx, types);
            if (!idx.empty()) {
                auto* op = ops[idx[0]].get();
                if (op->type == MNN::OpType_BinaryOp) {
                    for (int inIdx : op->inputIndexes) {
                        auto* blb = constBlobOf(ops, inIdx);
                        if (blb && blb->float32s.size() >= 1) { str = blb->float32s[0]; break; }
                    }
                } else if (op->type == MNN::OpType_Convolution) {
                    auto* c = op->main.AsConvolution2D();
                    if (c && !c->weight.empty()) str = c->weight[0];
                }
            }
            std::vector<float> u = {float(mW), float(mH), str, 0, 0,0,0,0};
            fprintf(stderr, "[EXACT-FIRST-MATCH] pat=%s op=%d at i=%d\n", pat.ispType, (int)ops[idx[0]]->type, idx[0]);
    return applyExact(ops, i, pat, mW, mH, u);
        }

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
        // EXACT MATCH: find ADD as direct consumer of MUL output via connectivity.
        if (ops[i]->type == MNN::OpType_BinaryOp &&
            isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) {
            int mulOut = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
            if (mulOut < 0) return false;
            int j = consumerOfBinOp(ops, mulOut, MNN::BinaryOpOperation_ADD);
            if (j < 0) return false;

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
        // EXACT MATCH: find ReLU6 as direct consumer of Sub output.
        int subOut = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
        if (subOut < 0) return false;
        int j = consumerOfType(ops, subOut, MNN::OpType_ReLU6);
        if (j < 0) return false;
        // Extract black_level from Sub's second input (Const)
        float bl = 0.0f;
        int subIn1 = inputTensorOf(ops[i].get(), 1);
        auto* blb = constBlobOf(ops, subIn1);
        if (blb && !blb->float32s.empty()) bl = blb->float32s[0];
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

    // R6b: Conv(1×1)+Clip → isp.display_clip (post-demosaic clamp)
    // Absorbs standalone Clip after a Conv into the Conv as display_clip.
    bool tryDisplayClip(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_ReLU6) return false;
        // EXACT MATCH: find Conv(1×1) producer via connectivity.
        if (ops[i]->inputIndexes.empty()) return false;
        int tensorIdx = traceTensor(ops[i]->inputIndexes[0], ops);
        int prov = opIndexProducerOf(ops, tensorIdx);
        if (prov < 0 || !ops[prov]) return false;
        if (ops[prov]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[prov]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 1 || c->common->kernelY != 1) return false;
        // Absorb: just remove the Clip, let the Conv stand alone.
        ops[i]->outputIndexes = ops[prov]->outputIndexes;
        ops[prov].reset();
        VLOG(2) << "[P1] R6b: absorbed Conv+Clip into standalone Conv at " << prov;
        return true;
    }

    // R2c: Conv(1×1,N→N)+Clip → absorb Clip (identity Conv clamp)
    // For cases where a 1×1 identity-like Conv has a Clip after it.
    bool tryConvClipIdentity(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 1 || c->common->kernelY != 1) return false;
        // EXACT MATCH: find ReLU6 as direct consumer of Conv output.
        int convOut = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
        if (convOut < 0) return false;
        int next = consumerOfType(ops, convOut, MNN::OpType_ReLU6);
        if (next < 0) return false;
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
        // EXACT MATCH: find Conv producer via connectivity.
        int tensorIdx = traceTensor(ops[i]->inputIndexes[0], ops);
        int prov = opIndexProducerOf(ops, tensorIdx);
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
        // EXACT MATCH: find Mul as direct consumer of Sub output.
        int subOut = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
        if (subOut < 0) return false;
        int j = consumerOfBinOp(ops, subOut, MNN::BinaryOpOperation_MUL);
        if (j < 0) return false;
        // Absorb: Sub takes Mul's output, Mul is removed
        ops[i]->outputIndexes = ops[j]->outputIndexes;
        ops[j].reset();
        VLOG(2) << "[P1] R3d: Sub+Mul fused at " << i << " mul at " << j;
        i = j;
        return true;
    }

    // R3e: Mul+Add (channel bias) → fuse into single dispatch
    // Common in FCS: Mul(gain) + Add(bias)
    bool tryMulAdd(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        // EXACT MATCH: find Add as direct consumer of Mul output.
        int mulOut = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
        if (mulOut < 0) return false;
        int j = consumerOfBinOp(ops, mulOut, MNN::BinaryOpOperation_ADD);
        if (j < 0) return false;
        // Absorb: Mul takes Add's output, Add is removed
        ops[i]->outputIndexes = ops[j]->outputIndexes;
        ops[j].reset();
        VLOG(2) << "[P1] R3e: Mul+Add fused at " << i << " add at " << j;
        i = j;
        return true;
    }

    // R6d: Mul+ReLU6 → absorb (white-level clamp after scale)
    // Covers: BayerWb non-identity gains, any Mul+Clip pattern.
    bool tryMulClip(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        // EXACT MATCH: find ReLU6 as direct consumer of Mul output.
        int mulOut = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
        if (mulOut < 0) return false;
        int j = consumerOfType(ops, mulOut, MNN::OpType_ReLU6);
        if (j < 0) return false;
        ops[i]->outputIndexes = ops[j]->outputIndexes;
        ops[j].reset();
        VLOG(2) << "[P1] R6d: Mul+ReLU6 absorbed at " << i;
        i = j;
        return true;
    }

    // R3f: Sub+Max+Min → fuse (BLC50 pattern: Sub(dark_frame)+Max(0)+Min(max))
    bool trySubMaxMin(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        // EXACT MATCH: find Max as direct consumer of Sub output, then Min as consumer of Max.
        int subOut = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
        if (subOut < 0) return false;
        int maxIdx = consumerOfBinOp(ops, subOut, MNN::BinaryOpOperation_MAX);
        if (maxIdx < 0) return false;
        int maxOut = ops[maxIdx]->outputIndexes.empty() ? -1 : ops[maxIdx]->outputIndexes[0];
        if (maxOut < 0) return false;
        int k = consumerOfBinOp(ops, maxOut, MNN::BinaryOpOperation_MIN);
        if (k < 0) return false;
        // Fuse all three: Sub takes Min's output
        ops[i]->outputIndexes = ops[k]->outputIndexes;
        ops[maxIdx].reset();
        ops[k].reset();
        VLOG(2) << "[P1] R3f: Sub+Max+Min fused at " << i;
        i = k;
        return true;
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
        // EXACT MATCH: the Sub must consume the Conv's output tensor directly.
        int convOut0 = ops[i]->outputIndexes.empty() ? -1 : ops[i]->outputIndexes[0];
        if (convOut0 < 0) return false;
        int j = opIndexProducerOf(ops, convOut0); // direct consumer of Conv output
        if (j >= 0 && ops[j] && ops[j]->type == MNN::OpType_BinaryOp &&
            isBinOp(ops[j].get(), MNN::BinaryOpOperation_SUB)) {
            // Absorb: Conv takes Sub's output
            ops[i]->outputIndexes = ops[j]->outputIndexes;
            ops[j].reset();
            VLOG(2) << "[P1] R4c: Conv+Sub fused at " << i;
            i = j;
            return true;
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

        // GUARD: reject if Clip/ReLU6 follows Add (that's saturation/unsharp, not LDCI)
        // LDCI ends with Add(input + scaled). Saturation/Unsharp have Clip(0,1) after.
        int addOut = ops[addIdx]->outputIndexes.empty() ? -1 : ops[addIdx]->outputIndexes[0];
        if (addOut >= 0) {
            int nextIdx = consumerOfType(ops, addOut, MNN::OpType_ReLU6);
            if (nextIdx >= 0 && ops[nextIdx]) {
                VLOG(2) << "[P1] R5: rejecting at " << i << " — Clip/ReLU6 after Add (not LDCI)";
                return false;
            }
        }
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

        // EXACT-MATCH: a bare Pow is NOT a display op — GammaBlock's
        // Pow(x, inv_gamma) and ToneBlock's Pow chains must NOT be stolen.
        // Require BOTH:
        //   (a) the exponent const ≈ 2.2 (display gamma, e.g. sRGB→output)
        //   (b) the Pow output feeds a uint8 conversion chain
        //       (Mul(255) → Cast), NOT a Clip/ReLU (gamma signature).
        float exponent = -1.0f;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[k]->main.AsBlob();
                        if (blb && blb->float32s.size() >= 1) exponent = blb->float32s[0];
                    }
                }
            }
        }
        if (exponent < 0 || fabsf(exponent - 2.2f) > 0.01f) return false;

        // Reject clip consumers — Pow→Clip/ReLU is GammaBlock, not display.
        for (int j = 0; j < (int)ops.size(); j++) {
            if (!ops[j] || j == i) continue;
            if (ops[j]->type == MNN::OpType_ReLU || ops[j]->type == MNN::OpType_ReLU6 ||
                ops[j]->type == MNN::OpType_Extra && ops[j]->main.AsExtra() &&
                ops[j]->main.AsExtra()->type == "Clip") {
                for (int inIdx : ops[j]->inputIndexes) {
                    for (int outIdx : ops[i]->outputIndexes) {
                        if (inIdx == outIdx) return false;
                    }
                }
            }
        }

        std::vector<float> u = {float(mW),float(mH),0,1,1, 2.2f, 0,0};

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.display";
        buildCommonAttrs(ex, mW, mH, u);
        addNamedFloats(ex, "display", {2.2f, 0.0f});
        setEngine(ex);
        addSpirv(ex, "isp.display");
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] R6: display at " << i;
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
        // GUARD: only match spatial means (dim=[2,3]), not channel means (dim=[1])
        // SaturationBlock uses ReduceMean(axes=[1]) → should NOT be ldci
        {
            auto* r = ops[i]->main.AsReductionParam();
            if (r && r->operation == MNN::ReductionType_MEAN) {
                if (!r->dim.empty()) {
                    std::set<int> d(r->dim.begin(), r->dim.end());
                    if (d.count(2) && d.count(3) && d.size() == 2) {
                        // spatial mean — OK for ldci
                    } else {
                        VLOG(2) << "[P1] R5: RustReduceLdci reject non-spatial at " << i;
                        return false;
                    }
                }
            }
        }
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

    // Rust DisplayBlock: detect Mul(scale≈1 scalar) → isp.display identity gamma.
    // Strictly requires a SCALAR Const (single float) so grid maps / gain maps
    // (e.g. LSC gain_grid [1,4,16,16] of 1.0s) are NOT misclassified.
    bool tryRustDisplay(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        // Check if one input is a scalar Const(≈1.0) — the identity scale
        bool hasIdentityScale = false;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int j = 0; j < (int)ops.size(); j++) {
                if (!ops[j] || ops[j]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[j]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[j]->main.AsBlob();
                        if (blb && blb->float32s.size() == 1 && std::abs(blb->float32s[0] - 1.0f) < 0.01f) {
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
                        // gain_map is large (>100 floats) AND NOT a 4-channel
                        // Bayer-quad grid (those belong to isp.lsc — Pattern C).
                        // Vignetting maps are 1ch or 3ch, typically [1,1,H,W].
                        if (blb && blb->float32s.size() > 100 &&
                            !(blb->dims.size() == 4 && blb->dims[1] == 4)) {
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
    // Pattern A: Mul(image, full_res_gain_map_Const) - 1 op
    // Pattern B: Resize(bilinear, small_gain_map_Const) → Mul(image, resized) - 2 ops
    //   (from cam_app / softisp LscBlock ONNX pattern)
    bool tryLsc(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;

        // --- Pattern A: direct Const gain map at full sensor resolution ---
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
            if (gainMapIdx >= 0) {
                ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
                auto* ex = new MNN::ExtraT(); ex->type = "isp.lsc"; ex->engine = "MNN";
                auto* blb = ops[gainMapIdx]->main.AsBlob();
                std::vector<float> u = {float(mW), float(mH)};
                if (blb) {
                    u.push_back(float(blb->dims[blb->dims.size() - 1])); // gw
                    u.push_back(float(blb->dims[blb->dims.size() - 2])); // gh
                    u.insert(u.end(), blb->float32s.begin(), blb->float32s.end());
                }
                buildCommonAttrs(ex, mW, mH, u);
                setEngine(ex); addSpirv(ex, "isp.lsc"); ops[i]->main.value = ex;
                VLOG(2) << "[P1] LSC: radial gain map at " << i; return true;
            }
        }

        // --- Pattern B: Resize(bilinear) + Mul  (cam_app / softisp LscBlock) ---
        for (int inIdx : ops[i]->inputIndexes) {
            // Find an Interp (Resize) feeding this Mul input
            int interpIdx = -1;
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Interp) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) { interpIdx = k; break; }
                }
                if (interpIdx >= 0) break;
            }
            if (interpIdx < 0) continue;
            auto* interp = ops[interpIdx]->main.AsInterp();
            if (!interp || interp->resizeType != 2) continue; // must be bilinear

            // The Interp's input must be a small Const radial gain map
            int smallGainIdx = -1;
            for (int inIdx2 : ops[interpIdx]->inputIndexes) {
                for (int k = 0; k < (int)ops.size(); k++) {
                    if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                    for (int outIdx : ops[k]->outputIndexes) {
                        if (outIdx == inIdx2) { smallGainIdx = k; break; }
                    }
                    if (smallGainIdx >= 0) break;
                }
                if (smallGainIdx >= 0) break;
            }
            if (smallGainIdx < 0) continue;
            auto* blb = ops[smallGainIdx]->main.AsBlob();
            if (!blb || blb->dims.size() < 2) continue;

            // Fuse Resize+Mul into isp.lsc — store the small gain map;
            // the ISP shader handles bilinear interpolation at runtime.
            ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
            auto* ex = new MNN::ExtraT(); ex->type = "isp.lsc"; ex->engine = "MNN";
            std::vector<float> u = {float(mW), float(mH)};
            if (blb) {
                u.push_back(float(blb->dims[blb->dims.size() - 1])); // gw
                u.push_back(float(blb->dims[blb->dims.size() - 2])); // gh
                u.insert(u.end(), blb->float32s.begin(), blb->float32s.end());
            }
            buildCommonAttrs(ex, mW, mH, u);
            setEngine(ex); addSpirv(ex, "isp.lsc"); ops[i]->main.value = ex;
            ops[interpIdx].reset(); // remove the Resize; the shader does interpolation
            VLOG(2) << "[P1] LSC: Resize(bilinear)+Mul at " << interpIdx << "->" << i;
            return true;
        }

        // --- Pattern C: direct Const grid map [1,C,gh,gw] at sub-resolution ---
        // Standalone LscBlock graph: Mul(input, gain_grid_tiled[1,4,H/2,W/2]) with
        // the grid as a Const (baked params). No Resize in the graph — the shader
        // samples the grid directly.
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx != inIdx) continue;
                    auto* blb = ops[k]->main.AsBlob();
                    if (!blb || blb->dims.size() != 4) continue;
                    int c = blb->dims[1] > 0 ? (int)blb->dims[1] : 0;
                    int gh = blb->dims[2] > 0 ? (int)blb->dims[2] : 0;
                    int gw = blb->dims[3] > 0 ? (int)blb->dims[3] : 0;
                    // Must be a 4-channel Bayer quad grid strictly smaller than
                    // the frame (else it is an identity display scale / other).
                    if (c != 4 || gh <= 0 || gw <= 0 || gh > mH || gw > mW) continue;
                    if (gh == mH && gw == mW) continue; // full-res → Pattern A domain
                    ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
                    auto* ex = new MNN::ExtraT(); ex->type = "isp.lsc"; ex->engine = "MNN";
                    std::vector<float> u = {float(mW), float(mH), float(gw), float(gh)};
                    u.insert(u.end(), blb->float32s.begin(), blb->float32s.end());
                    buildCommonAttrs(ex, mW, mH, u);
                    setEngine(ex); addSpirv(ex, "isp.lsc"); ops[i]->main.value = ex;
                    VLOG(2) << "[P1] LSC: direct grid [1,4," << gh << "," << gw << "] at " << i;
                    return true;
                }
            }
        }
        return false;
    }

    // AWB (Auto White Balance)
    bool tryAwb(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_MUL)) return false;
        float gains[3] = {1,1,1}; int gainConstIdx = -1;
        // EXACT-MATCH: find the per-channel gain const among ALL inputs.
        // WbGainsBlock is Mul(data, gains[1,3,1,1]) — NO ADD chain. The
        // older code bailed if the FIRST input wasn't the const (frame is a
        // tensor, not const) and required a downstream ADD. Fix both.
        for (int inIdx : ops[i]->inputIndexes) {
            if (gainConstIdx >= 0) break;
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[k]->main.AsBlob();
                        if (blb && blb->float32s.size() >= 3) {
                            for (int c=0;c<3;c++) gains[c]=blb->float32s[c];
                            gainConstIdx=k; break;
                        }
                    }
                }
                if (gainConstIdx >= 0) break;
            }
        }
        if (gainConstIdx < 0) return false;

        // Optional downstream ADD (Mul→Add chains like AlgoAwbBlock):
        // offsets come from the ADD's const input; absent → zeros.
        int addIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j] || ops[j]->type != MNN::OpType_BinaryOp) continue;
            if (isBinaryType(ops[j].get(), MNN::BinaryOpOperation_ADD) && isChain(ops[i].get(), ops[j].get())) { addIdx = j; break; }
        }
        float offsets[3] = {0,0,0};
        if (addIdx >= 0) {
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
        }

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.awb"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), gains[0], gains[1], gains[2], offsets[0], offsets[1], offsets[2]};
        buildCommonAttrs(ex, mW, mH, u); addNamedFloats(ex, "awb", {gains[0], gains[1], gains[2], offsets[0], offsets[1], offsets[2]});
        setEngine(ex); addSpirv(ex, "isp.awb"); ops[i]->main.value = ex;
        // Drop the gain const input — only the data tensor remains (shader reads gains from const buffer).
        if (gainConstIdx >= 0) {
            auto& inIdx = ops[i]->inputIndexes;
            for (int outIdx : ops[gainConstIdx]->outputIndexes) {
                for (auto it = inIdx.begin(); it != inIdx.end(); ++it) {
                    if (*it == outIdx) { inIdx.erase(it); break; }
                }
            }
        }
        if (gainConstIdx >= 0) ops[gainConstIdx].reset(); if (addIdx >= 0) ops[addIdx].reset();
        VLOG(2) << "[P1] AWB: channel gains at " << i; return true;
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
                ops[i]->main.value = ex;
                // Drop the gain const input — only the data tensor remains.
                if (gainConstIdx >= 0) {
                    auto& inIdx = ops[i]->inputIndexes;
                    for (int outIdx : ops[gainConstIdx]->outputIndexes) {
                        for (auto it = inIdx.begin(); it != inIdx.end(); ++it) {
                            if (*it == outIdx) { inIdx.erase(it); break; }
                        }
                    }
                }
                if (gainConstIdx >= 0) ops[gainConstIdx].reset(); if (addIdx >= 0) ops[addIdx].reset();
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
    // EXACT MATCH: calibration quad stats. Only fuse a Reduction when:
    //   - operation is MEAN / MIN / MAX
    //   - axes are exactly {2,3} (spatial H×W) with keepDims=1
    //   - input tensor is a full frame, NOT an already-reduced [..,1,1] tensor
    // This excludes channel reductions (Saturation/ColorSpace axes=[1]) and
    // keepdims=0 stats (StatsBlock/AlgoAwbBlock) which have different shaders.
    bool tryCalibStats(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Reduction) return false;
        auto* red = ops[i]->main.AsReductionParam(); if (!red) return false;
        if (red->operation != MNN::ReductionType_MEAN && red->operation != MNN::ReductionType_MIN && red->operation != MNN::ReductionType_MAX) return false;
        // axes must be exactly {2,3} (spatial) — never channel collapse {1}
        if (red->dim.size() != 2) return false;
        if (red->dim[0] != 2 || red->dim[1] != 3) return false;
        if (!red->keepDims) return false;
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.calib_stats"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), (float)red->operation};
        buildReduceAttrs(ex, u); setEngine(ex); addSpirv(ex, "isp.calib_stats");
        // Drop the axes const input — the shader reduces over all spatial dims.
        {
            auto& inIdx = ops[i]->inputIndexes;
            for (auto it = inIdx.begin(); it != inIdx.end();) {
                bool isConst = false;
                for (int k = 0; k < (int)ops.size(); k++) {
                    if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                    for (int outIdx : ops[k]->outputIndexes) {
                        if (outIdx == *it) { isConst = true; break; }
                    }
                    if (isConst) break;
                }
                if (isConst) it = inIdx.erase(it); else ++it;
            }
        }
        ops[i]->main.value = ex; VLOG(2) << "[P1] CalibStats: op=" << (int)red->operation << " at " << i; return true;
    }

    // IspController Stats
    // EXACT MATCH: controller stats. Only fuse a ReduceMean when axes are
    // exactly {2,3} with keepDims=1 (spatial mean of a full frame). Channel
    // reductions (axes=[1]) and keepdims=0 stats use different shaders.
    bool tryIspControllerStats(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Reduction) return false;
        auto* red = ops[i]->main.AsReductionParam(); if (!red) return false;
        if (red->operation != MNN::ReductionType_MEAN) return false;
        if (red->dim.size() != 2) return false;
        if (red->dim[0] != 2 || red->dim[1] != 3) return false;
        if (!red->keepDims) return false;
        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.ispc_stats"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH)};
        buildReduceAttrs(ex, u); setEngine(ex); addSpirv(ex, "isp.ispc_stats");
        // Drop the axes const input — the shader reduces over all spatial dims.
        {
            auto& inIdx = ops[i]->inputIndexes;
            for (auto it = inIdx.begin(); it != inIdx.end();) {
                bool isConst = false;
                for (int k = 0; k < (int)ops.size(); k++) {
                    if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                    for (int outIdx : ops[k]->outputIndexes) {
                        if (outIdx == *it) { isConst = true; break; }
                    }
                    if (isConst) break;
                }
                if (isConst) it = inIdx.erase(it); else ++it;
            }
        }
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
        std::vector<float> u={float(mW),float(mH)};
        buildReduceAttrs(ex,u); setEngine(ex); addSpirv(ex,"isp.af_focus");
        // Drop axes const inputs on the final reduction — shader reduces all spatial dims.
        {
            auto& inIdx = ops[red2Idx]->inputIndexes;
            for (auto it = inIdx.begin(); it != inIdx.end();) {
                bool isConst = false;
                for (int k = 0; k < (int)ops.size(); k++) {
                    if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                    for (int outIdx : ops[k]->outputIndexes) {
                        if (outIdx == *it) { isConst = true; break; }
                    }
                    if (isConst) break;
                }
                if (isConst) it = inIdx.erase(it); else ++it;
            }
        }
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


    // ════════════════════════════════════════════════════════════════════
    //  cam_app ISP block patterns
    // ════════════════════════════════════════════════════════════════════

    // Unsharp: AvgPool(3x3)+Sub+Mul+Add+Clip -> isp.fcs (reuse fcs SPIR-V)
    bool tryUnsharp(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Pooling) return false;
        auto* p = ops[i]->main.AsPool();
        if (!p || p->type != MNN::PoolType_AVEPOOL) return false;
        int subIdx = -1, mulIdx = -1, addIdx = -1, clipIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 8); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp) {
                auto* bin = ops[j]->main.AsBinaryOp();
                if (bin->opType == MNN::BinaryOpOperation_SUB && subIdx < 0 && isChainSkipCT(ops[i].get(), ops[j].get(), ops)) subIdx = j;
                else if (bin->opType == MNN::BinaryOpOperation_MUL && mulIdx < 0 && subIdx >= 0 && isChainSkipCT(ops[subIdx].get(), ops[j].get(), ops)) mulIdx = j;
                else if (bin->opType == MNN::BinaryOpOperation_ADD && addIdx < 0 && mulIdx >= 0 && isChainSkipCT(ops[mulIdx].get(), ops[j].get(), ops)) addIdx = j;
            } else if (ops[j]->type == MNN::OpType_ReLU6 && addIdx >= 0 && clipIdx < 0) {
                clipIdx = j;
            }
        }
        if (subIdx < 0 || mulIdx < 0 || addIdx < 0) return false;
        int lastIdx = clipIdx >= 0 ? clipIdx : addIdx;
        ops[lastIdx]->type = MNN::OpType_Extra;
        ops[lastIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), 1.0f};
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex); addSpirv(ex, "isp.fcs");
        ops[lastIdx]->main.value = ex;
        if (subIdx >= 0) ops[subIdx].reset();
        if (mulIdx >= 0 && mulIdx != lastIdx) ops[mulIdx].reset();
        if (addIdx >= 0 && addIdx != lastIdx) ops[addIdx].reset();
        if (clipIdx >= 0 && clipIdx != lastIdx) ops[clipIdx].reset();
        VLOG(2) << "[P1] Unsharp->fcs at " << i; i = lastIdx; return true;
    }

    // Saturation: ReduceMean+Sub+Mul+Add+Clip -> isp.fcs
    bool trySaturation(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Reduction) return false;
        auto* r = ops[i]->main.AsReductionParam();
        if (!r || r->operation != MNN::ReductionType_MEAN) return false;
        int subIdx = -1, mulIdx = -1, addIdx = -1, clipIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 8); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp) {
                auto* bin = ops[j]->main.AsBinaryOp();
                if (bin->opType == MNN::BinaryOpOperation_SUB && subIdx < 0 && isChainSkipCT(ops[i].get(), ops[j].get(), ops)) subIdx = j;
                else if (bin->opType == MNN::BinaryOpOperation_MUL && mulIdx < 0 && subIdx >= 0 && isChainSkipCT(ops[subIdx].get(), ops[j].get(), ops)) mulIdx = j;
                else if (bin->opType == MNN::BinaryOpOperation_ADD && addIdx < 0 && mulIdx >= 0 && isChainSkipCT(ops[mulIdx].get(), ops[j].get(), ops)) addIdx = j;
            } else if (ops[j]->type == MNN::OpType_ReLU6 && addIdx >= 0 && clipIdx < 0) {
                clipIdx = j;
            }
        }
        if (subIdx < 0 || mulIdx < 0 || addIdx < 0) return false;
        int lastIdx = clipIdx >= 0 ? clipIdx : addIdx;
        ops[lastIdx]->type = MNN::OpType_Extra;
        ops[lastIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), 1.0f};
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex); addSpirv(ex, "isp.fcs");
        ops[lastIdx]->main.value = ex;
        if (subIdx >= 0) ops[subIdx].reset();
        if (mulIdx >= 0 && mulIdx != lastIdx) ops[mulIdx].reset();
        if (addIdx >= 0 && addIdx != lastIdx) ops[addIdx].reset();
        if (clipIdx >= 0 && clipIdx != lastIdx) ops[clipIdx].reset();
        VLOG(2) << "[P1] Saturation->fcs at " << i; i = lastIdx; return true;
    }

    // BadPixel: Sub(zero,input)+MaxPool(3x3,MAX)+... skip fusion (no single SPIR-V)
    bool tryBadPixel(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        bool hasZero = false;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) { hasZero = true; break; }
                }
                if (hasZero) break;
            }
            if (hasZero) break;
        }
        if (!hasZero) return false;
        int poolIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
            if (!ops[j] || ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type == MNN::OpType_Pooling) {
                auto* p = ops[j]->main.AsPool();
                if (p && p->type == MNN::PoolType_MAXPOOL && p->kernelX == 3 && p->kernelY == 3) poolIdx = j;
            }
            break;
        }
        if (poolIdx < 0) return false;
        i = poolIdx;
        return false;
    }

    // BayerWb: Conv+Mul(Const gains) -> isp.fcs
    bool tryBayerWb(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        int mulIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp) {
                auto* bin = ops[j]->main.AsBinaryOp();
                if (bin->opType == MNN::BinaryOpOperation_MUL && isChainSkipCT(ops[i].get(), ops[j].get(), ops)) { mulIdx = j; break; }
            }
        }
        if (mulIdx < 0) return false;
        bool hasGains = false;
        for (int inIdx : ops[mulIdx]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) { hasGains = true; break; }
                }
                if (hasGains) break;
            }
            if (hasGains) break;
        }
        if (!hasGains) return false;
        ops[mulIdx]->type = MNN::OpType_Extra;
        ops[mulIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), 1.0f};
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex); addSpirv(ex, "isp.fcs");
        ops[mulIdx]->main.value = ex;
        ops[i].reset();
        VLOG(2) << "[P1] BayerWb->fcs at " << i; i = mulIdx; return true;
    }

    // Bilateral: Conv+Conv+Sub+Mul+Add+Clip -> isp.bilateral
    bool tryBilateral(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Convolution) return false;
        int conv2Idx = -1, subIdx = -1, mulIdx = -1, addIdx = -1, clipIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 8); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Convolution && conv2Idx < 0 && isChainSkipCT(ops[i].get(), ops[j].get(), ops)) {
                conv2Idx = j;
            } else if (ops[j]->type == MNN::OpType_BinaryOp) {
                auto* bin = ops[j]->main.AsBinaryOp();
                if (bin->opType == MNN::BinaryOpOperation_SUB && subIdx < 0 && conv2Idx >= 0 && isChainSkipCT(ops[conv2Idx].get(), ops[j].get(), ops)) subIdx = j;
                else if (bin->opType == MNN::BinaryOpOperation_MUL && mulIdx < 0 && subIdx >= 0 && isChainSkipCT(ops[subIdx].get(), ops[j].get(), ops)) mulIdx = j;
                else if (bin->opType == MNN::BinaryOpOperation_ADD && addIdx < 0 && mulIdx >= 0 && isChainSkipCT(ops[mulIdx].get(), ops[j].get(), ops)) addIdx = j;
            } else if (ops[j]->type == MNN::OpType_ReLU6 && addIdx >= 0 && clipIdx < 0) {
                clipIdx = j;
            }
        }
        if (conv2Idx < 0 || subIdx < 0 || mulIdx < 0 || addIdx < 0) return false;
        int lastIdx = clipIdx >= 0 ? clipIdx : addIdx;
        ops[lastIdx]->type = MNN::OpType_Extra;
        ops[lastIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.bilateral"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), 0.5f};
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex); addSpirv(ex, "isp.bilateral");
        ops[lastIdx]->main.value = ex;
        if (conv2Idx >= 0) ops[conv2Idx].reset();
        if (subIdx >= 0) ops[subIdx].reset();
        if (mulIdx >= 0 && mulIdx != lastIdx) ops[mulIdx].reset();
        if (addIdx >= 0 && addIdx != lastIdx) ops[addIdx].reset();
        if (clipIdx >= 0 && clipIdx != lastIdx) ops[clipIdx].reset();
        VLOG(2) << "[P1] Bilateral at " << i; i = lastIdx; return true;
    }

    // YuvSat: Sub+Mul+Sub+Add+Add+Clip -> isp.fcs
    bool tryYuvSat(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        int mulIdx = -1, sub2Idx = -1, add1Idx = -1, add2Idx = -1, clipIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 8); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp) {
                auto* bin = ops[j]->main.AsBinaryOp();
                if (bin->opType == MNN::BinaryOpOperation_MUL && mulIdx < 0 && isChainSkipCT(ops[i].get(), ops[j].get(), ops)) mulIdx = j;
                else if (bin->opType == MNN::BinaryOpOperation_SUB && sub2Idx < 0 && mulIdx >= 0 && isChainSkipCT(ops[mulIdx].get(), ops[j].get(), ops)) sub2Idx = j;
                else if (bin->opType == MNN::BinaryOpOperation_ADD && add1Idx < 0 && sub2Idx >= 0 && isChainSkipCT(ops[sub2Idx].get(), ops[j].get(), ops)) add1Idx = j;
                else if (bin->opType == MNN::BinaryOpOperation_ADD && add2Idx < 0 && add1Idx >= 0 && isChainSkipCT(ops[add1Idx].get(), ops[j].get(), ops)) add2Idx = j;
            } else if (ops[j]->type == MNN::OpType_ReLU6 && add2Idx >= 0 && clipIdx < 0) {
                clipIdx = j;
            }
        }
        if (mulIdx < 0 || sub2Idx < 0 || add1Idx < 0 || add2Idx < 0) return false;
        int lastIdx = clipIdx >= 0 ? clipIdx : add2Idx;
        ops[lastIdx]->type = MNN::OpType_Extra;
        ops[lastIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), 1.0f};
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex); addSpirv(ex, "isp.fcs");
        ops[lastIdx]->main.value = ex;
        if (mulIdx >= 0) ops[mulIdx].reset();
        if (sub2Idx >= 0) ops[sub2Idx].reset();
        if (add1Idx >= 0 && add1Idx != lastIdx) ops[add1Idx].reset();
        if (add2Idx >= 0 && add2Idx != lastIdx) ops[add2Idx].reset();
        if (clipIdx >= 0 && clipIdx != lastIdx) ops[clipIdx].reset();
        VLOG(2) << "[P1] YuvSat->fcs at " << i; i = lastIdx; return true;
    }

    // BLC: standalone Sub(zero, input) -> isp.blc (1 op)
    bool tryBLC(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        // Must have a Const input (black level value)
        bool hasConst = false;
        float bl = 0.0f;
        for (int inIdx : ops[i]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx) {
                        auto* blb = ops[k]->main.AsBlob();
                        if (blb && !blb->float32s.empty()) { bl = blb->float32s[0]; hasConst = true; }
                    }
                }
            }
        }
        if (!hasConst) return false;
        // Must NOT be followed by ReLU6 (that's trySubClipNormalize's job)
        for (int j = i + 1; j < std::min((int)ops.size(), i + 3); j++) {
            if (!ops[j] || ops[j]->type == MNN::OpType_Const) continue;
            if (ops[j]->type == MNN::OpType_ReLU6) return false; // let trySubClipNormalize handle it
            break;
        }
        ops[i]->type = MNN::OpType_Extra;
        ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), 1.0f, bl, 0,0,0,0};
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex); addSpirv(ex, "isp.fcs");
        addNamedFloats(ex, "fcs", {1.0f, bl});
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] BLC->fcs at " << i << " bl=" << bl;
        return true;
    }

    // Normalize: Cast(Int32→Float)+Div(sensor_max) -> isp.fcs (2 ops)
    bool tryNormalize(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Cast) return false;
        int divIdx = -1;
        for (int j = i + 1; j < std::min((int)ops.size(), i + 4); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp) {
                auto* bin = ops[j]->main.AsBinaryOp();
                if (bin->opType == MNN::BinaryOpOperation_DIV && isChainSkipCT(ops[i].get(), ops[j].get(), ops)) {
                    divIdx = j; break;
                }
            }
        }
        if (divIdx < 0) return false;
        float scale = 1.0f;
        for (int inIdx : ops[divIdx]->inputIndexes) {
            for (int k = 0; k < (int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type != MNN::OpType_Const) continue;
                for (int outIdx : ops[k]->outputIndexes) {
                    if (outIdx == inIdx && outIdx != ops[divIdx]->inputIndexes[0]) {
                        auto* blb = ops[k]->main.AsBlob();
                        if (blb && !blb->float32s.empty() && blb->float32s[0] > 0) scale = blb->float32s[0];
                    }
                }
            }
        }
        ops[divIdx]->type = MNN::OpType_Extra;
        ops[divIdx]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.fcs"; ex->engine = "MNN";
        std::vector<float> u = {float(mW), float(mH), 1.0f / scale};
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex); addSpirv(ex, "isp.fcs");
        addNamedFloats(ex, "fcs", {1.0f / scale, 0.0f});
        ops[divIdx]->main.value = ex;
        ops[i].reset(); // remove Cast
        VLOG(2) << "[P1] Normalize->fcs at " << i << " scale=" << scale;
        i = divIdx;
        return true;
    }

    // YuvSat7: Sub+Mul+Mul+Sub+Add+Add+Clip -> isp.fcs (7 ops)
    // Actual YuvSatBlock ONNX pattern has an extra Mul vs tryYuvSat's 6-op pattern.
    bool tryYuvSat7(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_BinaryOp) return false;
        if (!isBinaryType(ops[i].get(), MNN::BinaryOpOperation_SUB)) return false;
        int mul1=-1, mul2=-1, sub2=-1, add1=-1, add2=-1, clip=-1;
        for (int j = i+1; j < std::min((int)ops.size(), i+10); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_BinaryOp) {
                auto* b = ops[j]->main.AsBinaryOp();
                if (b->opType==MNN::BinaryOpOperation_MUL && mul1<0 && isChainSkipCT(ops[i].get(),ops[j].get(),ops)) mul1=j;
                else if (b->opType==MNN::BinaryOpOperation_MUL && mul2<0 && mul1>=0 && isChainSkipCT(ops[mul1].get(),ops[j].get(),ops)) mul2=j;
                else if (b->opType==MNN::BinaryOpOperation_SUB && sub2<0 && mul2>=0 && isChainSkipCT(ops[mul2].get(),ops[j].get(),ops)) sub2=j;
                else if (b->opType==MNN::BinaryOpOperation_ADD && add1<0 && sub2>=0 && isChainSkipCT(ops[sub2].get(),ops[j].get(),ops)) add1=j;
                else if (b->opType==MNN::BinaryOpOperation_ADD && add2<0 && add1>=0 && isChainSkipCT(ops[add1].get(),ops[j].get(),ops)) add2=j;
            } else if (ops[j]->type==MNN::OpType_ReLU6 && add2>=0 && clip<0) clip=j;
        }
        if (mul1<0 || sub2<0 || add1<0 || add2<0) return false;
        int last = clip>=0 ? clip : add2;
        ops[last]->type=MNN::OpType_Extra; ops[last]->main.type=MNN::OpParameter_Extra;
        auto* ex=new MNN::ExtraT(); ex->type="isp.fcs"; ex->engine="MNN";
        std::vector<float> u={float(mW),float(mH),1.0f};
        buildCommonAttrs(ex,mW,mH,u); setEngine(ex); addSpirv(ex,"isp.fcs");
        ops[last]->main.value=ex;
        if(mul1>=0) ops[mul1].reset(); if(mul2>=0) ops[mul2].reset();
        if(sub2>=0) ops[sub2].reset();
        if(add1>=0 && add1!=last) ops[add1].reset();
        if(add2>=0 && add2!=last) ops[add2].reset();
        if(clip>=0 && clip!=last) ops[clip].reset();
        VLOG(2) << "[P1] YuvSat7->fcs at " << i; i=last; return true;
    }

    // BayerWbReshape: Reshape->Conv->Reshape->Mul -> isp.fcs
    // Skips Reshape ops to match standalone BayerWbBlock pattern.
    bool tryBayerWbReshape(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (!ops[i] || ops[i]->type != MNN::OpType_Reshape) return false;
        // Skip Reshape -> Conv -> Reshape -> Mul
        int convIdx=-1, reshape2Idx=-1, mulIdx=-1;
        for (int j=i+1; j<std::min((int)ops.size(), i+6); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type==MNN::OpType_Reshape && convIdx<0) continue; // skip first reshape? no, this is after i
            if (ops[j]->type==MNN::OpType_Convolution && convIdx<0) { convIdx=j; continue; }
            if (ops[j]->type==MNN::OpType_Reshape && convIdx>=0 && reshape2Idx<0) { reshape2Idx=j; continue; }
            if (ops[j]->type==MNN::OpType_BinaryOp) {
                auto* b=ops[j]->main.AsBinaryOp();
                if (b->opType==MNN::BinaryOpOperation_MUL && convIdx>=0) { mulIdx=j; break; }
            }
        }
        if (convIdx<0 || mulIdx<0) return false;
        bool hasGains=false;
        for (int inIdx : ops[mulIdx]->inputIndexes) {
            for (int k=0; k<(int)ops.size(); k++) {
                if (!ops[k] || ops[k]->type!=MNN::OpType_Const) continue;
                for (int out : ops[k]->outputIndexes) { if (out==inIdx) { hasGains=true; break; } }
                if (hasGains) break;
            }
            if (hasGains) break;
        }
        if (!hasGains) return false;
        ops[mulIdx]->type=MNN::OpType_Extra; ops[mulIdx]->main.type=MNN::OpParameter_Extra;
        auto* ex=new MNN::ExtraT(); ex->type="isp.fcs"; ex->engine="MNN";
        std::vector<float> u={float(mW),float(mH),1.0f};
        buildCommonAttrs(ex,mW,mH,u); setEngine(ex); addSpirv(ex,"isp.fcs");
        ops[mulIdx]->main.value=ex;
        ops[i].reset(); // remove first Reshape
        if (convIdx>=0) ops[convIdx].reset();
        if (reshape2Idx>=0) ops[reshape2Idx].reset();
        VLOG(2) << "[P1] BayerWbReshape->fcs at " << i; i=mulIdx; return true;
    }

    // DemosaicStandalone: Conv(1x1,4->3ch) without backward BLC scan.
    // Matches standalone DebayerBlock/EdgeDemosaicBlock where the Conv
    // stands alone without preceding Div/Sub/Clip ops.
    bool tryDemosaicStandalone(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!c || !c->common) return false;
        if (c->common->kernelX != 1 || c->common->kernelY != 1) return false;
        if (c->common->outputCount != 3) return false;
        // Must NOT have a Const input from a preceding Div/Sub (that's tryDemosaic's job)
        // Simple check: if weight vector is present and has 12+ floats, it's a standalone 4->3 CCM
        if ((int)c->weight.size() < 12) return false;
        // Extract CCM from weights
        std::vector<float> ccm = {1,0,0, 0,1,0, 0,0,1};
        const auto& w = c->weight;
        for (int oc=0; oc<3; oc++) {
            ccm[oc*3+0] = w[oc*4+0];
            ccm[oc*3+1] = 2.0f*w[oc*4+1];
            ccm[oc*3+2] = w[oc*4+3];
        }
        ops[i]->type=MNN::OpType_Extra; ops[i]->main.type=MNN::OpParameter_Extra;
        auto* ex=new MNN::ExtraT(); ex->type="isp.demosaic_ccm"; ex->engine="MNN";
        std::vector<float> u={float(mW),float(mH),float(mInW),float(mInH),1023.0f,
            ccm[0],ccm[1],ccm[2],ccm[3],ccm[4],ccm[5],ccm[6],ccm[7],ccm[8],0,0,0,0};
        buildCommonAttrs(ex,mW,mH,u);
        addNamedFloats(ex,"ccm",{ccm[0],ccm[1],ccm[2],ccm[3],ccm[4],ccm[5],ccm[6],ccm[7],ccm[8]});
        setEngine(ex); addSpirv(ex,"isp.demosaic_ccm");
        ops[i]->main.value=ex;
        VLOG(2) << "[P1] DemosaicStandalone->demosaic_ccm at " << i;
        return true;
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
        clearElementwise(ops[i]->main.AsExtra());
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
        clearElementwise(ops[i]->main.AsExtra());
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
        // ── Check ONNX metadata for ISP enable flag ──
        // ISP fusion is OFF by default. To enable, the ONNX source must
        // have metadata_props { "isp_fusion", "enable" }.
        //
        // This keeps the converter safe: without the flag, it produces
        // primitive ops (Conv, BinaryOp, etc.) that all backends handle.
        // With the flag, ISP fusion runs and embeds SPIR-V into Extra ops.
        //
        // Kotlin API: MnnConvert.convert(onnxBytes, ispFusion = true)
        // JNI path:   OnnxPreprocessor.injectMetadata(onnx, "isp_fusion", "enable")
        {
            bool enableIsp = false;
            // PRIMARY: modelConfig metadata field (propagated from ONNX
            // metadata_props "isp_fusion" in cli.cpp before optimizeNet).
            // This is the runtime-switchable metadata path — reliable because
            // it rides on the modelConfig object that survives all passes.
            auto* cfg = Global<modelConfig>::Get();
            if (cfg != nullptr && cfg->ispFusionMeta.size() >= 6 &&
                cfg->ispFusionMeta.substr(0, 6) == "enable") {
                enableIsp = true;
            }
            // Fallback A: ExtraInfo buffer (ONNX metadata_props stored here as ExtraT flatbuffer)
            if (!enableIsp && net->extraInfo && !net->extraInfo->buffer.empty()) {
                auto* exInfo = flatbuffers::GetRoot<MNN::Extra>(net->extraInfo->buffer.data());
                if (exInfo && exInfo->attr()) {
                    for (auto* a : *exInfo->attr()) {
                        if (a && a->key() && a->key()->str() == "isp_fusion" &&
                            a->s() && a->s()->str().substr(0, 6) == "enable") {
                            enableIsp = true;
                            break;
                        }
                    }
                }
            }
            // Also check first op if it's a Meta Extra op
            if (!enableIsp && !net->oplists.empty()) {
                auto& first = net->oplists[0];
                if (first && first->type == MNN::OpType_Extra) {
                    auto* ex = first->main.AsExtra();
                    if (ex && ex->type == "Meta") {
                        for (auto& a : ex->attr) {
                            if (a && a->key == "isp_fusion" && a->s.size() >= 6 &&
                                a->s.substr(0, 6) == "enable") {
                                enableIsp = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (!enableIsp) {
                fprintf(stderr, "[IspFusion] DISABLED (default). Pass isp_fusion=enable metadata to activate.\n");
                return true;  // skip fusion, pass through as-is
            }
            fprintf(stderr, "[IspFusion] ENABLED via ONNX metadata (isp_fusion=enable).\n");
        }

        VLOG(1) << "[IspFusion] === Pre-pass: Remove Identity ops ===";
        RemoveIdentityOps().onExecute(net);

        VLOG(1) << "[IspFusion] === Pass 1: Standard → ISP Extra ops ===";
        Pass1_ToExtra::instance()->onExecute(net);

        VLOG(1) << "[IspFusion] === Pass 2: MACRO-FUSION — Extra chain → fused Extra ===";
        Pass2_FuseExtra::instance()->onExecute(net);

        VLOG(1) << "[IspFusion] Complete: " << net->oplists.size() << " ops";

        // ═══ DIAGNOSTIC: Warn if primitive ops remain after ISP fusion ═══
        // After Pass1+Pass2, ISP-pattern ops should be Extra(isp.*) types.
        // Any remaining Conv/BinaryOp/UnaryOp/Pooling/Reduction may indicate
        // incomplete fusion or non-ISP ops.
        {
            int ispCount = 0, primCount = 0;
            std::map<std::string, int> ispTypes;
            std::map<MNN::OpType, int> primTypes;
            for (size_t i = 0; i < net->oplists.size(); i++) {
                auto& op = net->oplists[i];
                if (!op) continue;
                if (op->type == MNN::OpType_Extra) {
                    auto* ex = op->main.AsExtra();
                    if (ex && ex->type.size() > 4 && ex->type.substr(0, 4) == "isp.") {
                        ispCount++;
                        ispTypes[ex->type]++;
                    }
                } else if (op->type == MNN::OpType_Convolution ||
                           op->type == MNN::OpType_BinaryOp ||
                           op->type == MNN::OpType_UnaryOp ||
                           op->type == MNN::OpType_Pooling ||
                           op->type == MNN::OpType_Reduction ||
                           op->type == MNN::OpType_Scale ||
                           op->type == MNN::OpType_ReLU ||
                           op->type == MNN::OpType_ReLU6 ||
                           op->type == MNN::OpType_Cast ||
                           op->type == MNN::OpType_Eltwise) {
                    primCount++;
                    primTypes[op->type]++;
                }
            }
            if (ispCount > 0) {
                fprintf(stderr, "[IspFusion] ISP opsets in use (%d total):\n", ispCount);
                for (auto& kv : ispTypes) {
                    fprintf(stderr, "  %-30s  x%d\n", kv.first.c_str(), kv.second);
                }
            }
            if (primCount > 0) {
                fprintf(stderr, "[IspFusion] WARNING: %d primitive ops remain after ISP fusion:\n", primCount);
                for (auto& kv : primTypes) {
                    fprintf(stderr, "  %-30s  x%d\n", MNN::EnumNamesOpType()[kv.first], kv.second);
                }
            } else {
                fprintf(stderr, "[IspFusion] OK: All ISP-pattern ops converted to Extra.\n");
            }
        }

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
    