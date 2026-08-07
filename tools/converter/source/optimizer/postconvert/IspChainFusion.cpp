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
//    │  R5.  Pool+Sub+Mul+Add               → isp.ldci           │
//    │  R5b. ReduceMean+Sub+Mul+Add+Clip    → isp.ldci_a (Rust)  │
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

// ── Log silencing ──────────────────────────────────────────────────────────
// logkit's VLOG(x) is NOT verbosity-gated — it expands to
// LogMessage(...).stream() and writes every diagnostic line to std::cout
// unconditionally. Fusion verification now uses the mnn2json opset (JVM
// side, see MnnJsonOpset.kt), so all in-TU diagnostic output is discarded
// into a null sink instead of spamming the converter's stdout/stderr.
class IspNullStreamBuf : public std::streambuf {
public:
    int_type overflow(int_type c) override { return c; }
};
static IspNullStreamBuf g_ispNullBuf;
static std::ostream g_ispVlogSink(&g_ispNullBuf);
#undef VLOG
#define VLOG(x) g_ispVlogSink

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
        {"isp.ldci_a",          g_ldci_a_spv,         g_ldci_a_spv_len},   // Rust ReduceMean LDCI variant
        {"isp.display",         g_display_spv,         g_display_spv_len},
        {"isp.fcs_display",     g_fcs_display_spv,     g_fcs_display_spv_len},
        {"isp.ee_ldci",         g_ee_ldci_spv,         g_ee_ldci_spv_len},
        {"isp.unpack_demosaic", g_unpack_demosaic_spv, g_unpack_demosaic_spv_len},
        {"isp.demosaic_interp", g_demosaic_interp_spv, g_demosaic_interp_spv_len},
        // Unified isp.demosaic opset — algorithm parameter selects SPIR-V
        {"isp.demosaic_binning", g_unpack_blc_spv,      g_unpack_blc_spv_len},
        {"isp.demosaic_bilinear", g_demosaic_interp_spv, g_demosaic_interp_spv_len},
        {"isp.demosaic_mhc",     g_demosaic_mhc_spv,     g_demosaic_mhc_spv_len},
        {"isp.demosaic_edge",   g_demosaic_spv,        g_demosaic_spv_len},
        {"isp.demosaic_a",      g_demosaic_spv,        g_demosaic_spv_len},
        {"isp.demosaic_debayer",g_demosaic_spv,        g_demosaic_spv_len},
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
    VLOG(2) << "[IspFusion] No SPIR-V for '" << type << "'";
#else
    VLOG(2) << "[IspFusion] MNN_ISP_EMBED_SPIRV not defined";
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
struct ChainConstCheck {
    int chainPos = 0;              // 0-based position in the collected chain
    int inputIdx = 0;              // which input of that op is the const
    std::vector<float> values;     // expected const values (1e-4 tol)
    ChainConstCheck(int p, int in, std::vector<float> v)
        : chainPos(p), inputIdx(in), values(std::move(v)) {}
};

struct ExactPattern {
    std::vector<MNN::OpType> opTypes;
    int constElems;
    int constIndex;
    const char* ispType;
    const char* spvName;
    const char* namedKey;
    int convWeightElems = -1;
    MNN::BinaryOpOperation binOpType = MNN::BinaryOpOperation_ADD; // default: any BinaryOp
    std::vector<float> convWeightValues;   // if non-empty, require weight match (1e-4 tol)
    std::vector<float> constValues;        // if non-empty, require const blob match (1e-4 tol)
    // noFuse: match and consume the chain (advance scan) but KEEP ops primitive.
    // Used as longest-match guard so shorter generic patterns (e.g. auto_contrast)
    // never steal scalar control chains (algo_gamma / algo_cct).
    bool noFuse = false;
    // Per-position const-value checks: (chainPos, inputIdx) -> expected values.
    std::vector<ChainConstCheck> chainConstChecks;
    // Profile-variant disambiguation (blocks whose constants are graph Inputs):
    //   inputTrace[k]      — required producer type of ops[idx[0]]->inputIndexes[k]
    //                        (traced through ConvertTensor; -1 = any). Stops at Extra.
    //   inputMustBeInput   — input indices that must be fed directly by a graph Input op.
    //   nextOpType         — required type of the next non-CT/Const/Input op after the chain.
    //   chainBinOps        — (chainPos, BinaryOp sub-type) requirements at chain positions.
    std::vector<int> inputTrace;
    std::vector<int> inputMustBeInput;
    int nextOpType = -1;
    int nextBinOp = -1;   // required BinaryOp sub-type of the next op (-1 = any)
    std::vector<std::pair<int, MNN::BinaryOpOperation>> chainBinOps;
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, const char* nk = nullptr, int cwe = -1,
                 std::vector<float> cwv = {}, std::vector<float> cv = {})
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), convWeightElems(cwe),
          convWeightValues(std::move(cwv)), constValues(std::move(cv)) {}
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, MNN::BinaryOpOperation bot,
                 const char* nk = nullptr, int cwe = -1,
                 std::vector<float> cwv = {}, std::vector<float> cv = {})
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), namedKey(nk), convWeightElems(cwe), binOpType(bot),
          convWeightValues(std::move(cwv)), constValues(std::move(cv)) {}
    // Guard constructor: longest-match no-fuse protection for scalar control chains.
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, bool nf,
                 std::vector<ChainConstCheck> ccc = {})
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), noFuse(nf), chainConstChecks(std::move(ccc)) {}
    // Profile-variant constructor: tolerates Input-as-const and adds structural
    // provenance disambiguation (inputTrace / inputMustBeInput / nextOpType).
    // The 7th arg is the bool marker so this ctor never collides with the
    // BinaryOp-sub-type ctor (whose 7th arg is the namedKey const char*).
    ExactPattern(std::vector<MNN::OpType> ops, int ce, int ci,
                 const char* isp, const char* spv, MNN::BinaryOpOperation bot,
                 bool pv, std::vector<int> itr, std::vector<int> imbi = {},
                 int nxt = -1, std::vector<std::pair<int, MNN::BinaryOpOperation>> cbo = {},
                 int nbo = -1)
        : opTypes(std::move(ops)), constElems(ce), constIndex(ci),
          ispType(isp), spvName(spv), binOpType(bot),
          inputTrace(std::move(itr)), inputMustBeInput(std::move(imbi)),
          nextOpType(nxt), nextBinOp(nbo), chainBinOps(std::move(cbo)) {
        (void)pv;
    }
};

// Find the op that produces tensorId, tracing through ConvertTensor to the
// ultimate producer. Returns MNN::OpType_Input for graph inputs, and stops
// at Extra ops (already-fused isp.* producers return Extra, not their origin).
static int producerTypeOf(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId) {
    for (const auto& op : ops) {
        if (!op) continue;
        for (int o : op->outputIndexes) {
            if (o != tensorId) continue;
            if (op->type == MNN::OpType_ConvertTensor && !op->inputIndexes.empty())
                return producerTypeOf(ops, op->inputIndexes[0]);
            return (int)op->type;
        }
    }
    return (int)MNN::OpType_Input; // not produced by any op = graph input
}

// Is tensorId fed directly by a graph Input op (no ConvertTensor wrapping)?
static bool isInputTensor(const std::vector<std::unique_ptr<OpT>>& ops, int tensorId) {
    for (const auto& op : ops) {
        if (!op || op->type != MNN::OpType_Input) continue;
        for (int o : op->outputIndexes) {
            if (o == tensorId) return true;
        }
    }
    return false;
}

// Navigate ops starting at , skipping Const and ConvertTensor,
// collecting up to  op types. Returns actual index per collected op.
static bool collectChain(const std::vector<std::unique_ptr<OpT>>& ops,
                         int start, int count,
                         std::vector<int>& indices, std::vector<MNN::OpType>& types) {
    int j = start;
    for (int n = 0; n < count && j < (int)ops.size(); j++) {
        if (!ops[j]) continue;
        if (ops[j]->type == MNN::OpType_Const ||
            ops[j]->type == MNN::OpType_ConvertTensor ||
            ops[j]->type == MNN::OpType_Input) continue;
        indices.push_back(j);
        types.push_back(ops[j]->type);
        n++;
    }
    return (int)types.size() == count;
}

// Check if an exact pattern matches at ops[i].
static bool matchExact(const std::vector<std::unique_ptr<OpT>>& ops,
                       int i, const ExactPattern& pat) {
    const bool dbg = (pat.inputMustBeInput.size() > 0 || pat.inputTrace.size() > 0 || pat.nextBinOp >= 0 || pat.chainBinOps.size() > 0);
    // Always log at VLOG(2) for key ISP blocks (demosaic_ccm, ee) to diagnose fusion failures.
    const bool keyBlock = (std::string(pat.ispType) == "isp.demosaic_ccm" ||
                           std::string(pat.ispType) == "isp.ee");
#define MDBG(msg) do { if (dbg) VLOG(1) << "[P1] DBG " << pat.ispType << "@" << i << " " << msg; } while(0)
#define KLOG(msg) do { if (keyBlock) VLOG(2) << "[P1] KEY " << pat.ispType << "@" << i << " " << msg; } while(0)
    if (i < 0 || i >= (int)ops.size() || !ops[i]) { MDBG("no-op"); KLOG("no-op"); return false; }
    std::vector<int> idx;
    std::vector<MNN::OpType> types;
    if (!collectChain(ops, i, (int)pat.opTypes.size(), idx, types)) {
        KLOG("collectChain-fail need=" << (int)pat.opTypes.size());
        MDBG("collectChain-fail"); return false;
    }
    for (int k = 0; k < (int)pat.opTypes.size(); k++) {
        if (types[k] != pat.opTypes[k]) {
            KLOG("optype-mismatch[" << k << "] got=" << MNN::EnumNameOpType(types[k])
                 << " want=" << MNN::EnumNameOpType(pat.opTypes[k]));
            MDBG("optype-mismatch"); return false;
        }
    }
    // Pass1 re-entrancy guard: never re-fuse an already-fused isp.* Extra.
    // The {Extra} pattern (GridSample→isp.warp) must only match raw ONNX
    // Extras, not isp.* ops produced by a previous conversion round (MNN→MNN).
    if (types[0] == MNN::OpType_Extra) {
        auto* ex = ops[idx[0]]->main.AsExtra();
        if (ex && ex->type.rfind("isp.", 0) == 0) { MDBG("reentrancy"); return false; }
    }
    // Check const constraint: inputIndexes[constIndex] must be a Const with constElems floats
    if (pat.constIndex >= 0 && pat.constElems >= 0) {
        if (idx.empty()) { MDBG("ce-empty"); return false; }
        auto* op = ops[idx[0]].get();
        if (pat.constIndex >= (int)op->inputIndexes.size()) { MDBG("ce-ci-oob"); return false; }
        int tensorId = op->inputIndexes[pat.constIndex];
        auto* blb = constBlobOf(ops, tensorId);
        if (!blb || (int)blb->float32s.size() != pat.constElems) { MDBG("ce-const-fail"); return false; }
    }
    // Check conv weight count — works for both Convolution and ConvolutionDepthwise
    // (both use Convolution2D flatbuffer layout with weight[]).
    if (pat.convWeightElems >= 0) {
        if (idx.empty()) { MDBG("cwe-empty"); return false; }
        auto* op = ops[idx[0]].get();
        if (op->type != MNN::OpType_Convolution && op->type != MNN::OpType_ConvolutionDepthwise) {
            KLOG("cwe-notconv got=" << MNN::EnumNameOpType(op->type));
            MDBG("cwe-notconv"); return false;
        }
        auto* c = op->main.AsConvolution2D();
        int actualW = c ? (int)c->weight.size() : -1;
        if (!c || actualW != pat.convWeightElems) {
            KLOG("cwe-count got=" << actualW << " want=" << pat.convWeightElems);
            MDBG("cwe-count"); return false;
        }
    }
    // Check conv weight VALUES (1e-4 tolerance) — disambiguates CCM identity vs
    // BT.601 CSC vs pyramid identity when op structures are identical.
    if (!pat.convWeightValues.empty()) {
        if (idx.empty()) return false;
        auto* op = ops[idx[0]].get();
        if (op->type != MNN::OpType_Convolution && op->type != MNN::OpType_ConvolutionDepthwise) return false;
        auto* c = op->main.AsConvolution2D();
        if (!c || (int)c->weight.size() != (int)pat.convWeightValues.size()) return false;
        for (size_t k = 0; k < pat.convWeightValues.size(); k++) {
            if (std::fabs(c->weight[k] - pat.convWeightValues[k]) > 1e-4f) return false;
        }
    }
    // Check const blob VALUES (1e-4 tolerance)
    if (!pat.constValues.empty()) {
        if (idx.empty()) return false;
        auto* op = ops[idx[0]].get();
        if (pat.constIndex < 0 || pat.constIndex >= (int)op->inputIndexes.size()) return false;
        int tensorId = op->inputIndexes[pat.constIndex];
        auto* blb = constBlobOf(ops, tensorId);
        if (!blb || (int)blb->float32s.size() != (int)pat.constValues.size()) return false;
        for (size_t k = 0; k < pat.constValues.size(); k++) {
            if (std::fabs(blb->float32s[k] - pat.constValues[k]) > 1e-4f) return false;
        }
    }
    // Check per-position const VALUES (1e-4 tolerance) — guards for scalar
    // control chains (algo_gamma / algo_cct) where consts sit at later chain
    // positions, not at idx[0].
    for (const auto& cc : pat.chainConstChecks) {
        if (cc.chainPos < 0 || cc.chainPos >= (int)idx.size()) return false;
        auto* op = ops[idx[cc.chainPos]].get();
        if (cc.inputIdx < 0 || cc.inputIdx >= (int)op->inputIndexes.size()) return false;
        int tensorId = op->inputIndexes[cc.inputIdx];
        auto* blb = constBlobOf(ops, tensorId);
        if (!blb || (int)blb->float32s.size() != (int)cc.values.size()) return false;
        for (size_t k = 0; k < cc.values.size(); k++) {
            if (std::fabs(blb->float32s[k] - cc.values[k]) > 1e-4f) return false;
        }
    }
    // Profile-variant input provenance: required producer types through CT.
    // (Inputs are legal producers — the block constants are graph Inputs.)
    if (!pat.inputTrace.empty()) {
        if (idx.empty()) { MDBG("itr-empty"); return false; }
        auto* op = ops[idx[0]].get();
        for (int k = 0; k < (int)pat.inputTrace.size(); k++) {
            int req = pat.inputTrace[k];
            if (req < 0) continue;
            if (k >= (int)op->inputIndexes.size()) { MDBG("itr-oob"); return false; }
            if (producerTypeOf(ops, op->inputIndexes[k]) != req) { MDBG("itr-producer"); return false; }
        }
    }
    // Profile-variant: inputs that must be fed directly by a graph Input op.
    for (int inIdx : pat.inputMustBeInput) {
        if (idx.empty()) { MDBG("imbi-empty"); return false; }
        auto* op = ops[idx[0]].get();
        if (inIdx < 0 || inIdx >= (int)op->inputIndexes.size()) { MDBG("imbi-oob"); return false; }
        if (!isInputTensor(ops, op->inputIndexes[inIdx])) { MDBG("imbi-notinput"); return false; }
    }
    // Profile-variant: required type of the next non-CT/Const/Input op.
    // Scan from the END of the matched chain (idx.back()+1), not the loop
    // index i — the chain may have skipped leading Input/CT anchors (e.g. a
    // 1-op anchored at i=27 with idx=[28]), in which case scanning from i+1
    // would hit the chain op itself and spuriously fail.
    if (pat.nextOpType >= 0) {
        bool found = false;
        int scanFrom = idx.empty() ? i : idx.back() + 1;
        for (int j = scanFrom; j < (int)ops.size(); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const ||
                ops[j]->type == MNN::OpType_ConvertTensor ||
                ops[j]->type == MNN::OpType_Input) continue;
            if ((int)ops[j]->type != pat.nextOpType) { MDBG("nxt-type"); return false; }
            found = true;
            break;
        }
        if (!found) { MDBG("nxt-notfound"); return false; }
    }
    // Profile-variant: required BinaryOp sub-type of the next op.
    // Disambiguates RawBlcBlock (SUB → next REALDIV = Normalize) from
    // BlcBlock (SUB → next MUL = Lsc/BayerWb) without tracing through
    // already-fused producers (crop/pyramid Extras).
    if (pat.nextBinOp >= 0) {
        bool found = false;
        int scanFrom = idx.empty() ? i : idx.back() + 1;
        for (int j = scanFrom; j < (int)ops.size(); j++) {
            if (!ops[j]) continue;
            if (ops[j]->type == MNN::OpType_Const ||
                ops[j]->type == MNN::OpType_ConvertTensor ||
                ops[j]->type == MNN::OpType_Input) continue;
            if (ops[j]->type != MNN::OpType_BinaryOp) { MDBG("nbo-notbinop"); return false; }
            auto* nb = ops[j]->main.AsBinaryOp();
            if (!nb || (int)nb->opType != pat.nextBinOp) { MDBG("nbo-optype"); return false; }
            found = true;
            break;
        }
        if (!found) { MDBG("nbo-notfound"); return false; }
    }
    // Per-position BinaryOp sub-type requirements (e.g. position 1→SUB, 2→MUL).
    for (const auto& cb : pat.chainBinOps) {
        if (cb.first < 0 || cb.first >= (int)idx.size()) return false;
        auto* op = ops[idx[cb.first]].get();
        if (op->type != MNN::OpType_BinaryOp) return false;
        auto* bin = op->main.AsBinaryOp();
        if (!bin || bin->opType != cb.second) return false;
    }
    // Check BinaryOp sub-type (Mul vs Div vs Add vs Sub)
    if (pat.binOpType != MNN::BinaryOpOperation_ADD) {
        if (idx.empty()) return false;
        auto* op = ops[idx[0]].get();
        if (op->type != MNN::OpType_BinaryOp) return false;
        auto* bin = op->main.AsBinaryOp();
        if (!bin || bin->opType != pat.binOpType) return false;
    }
    KLOG("MATCH");
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
    // Stride-2 ops (pyramid / demosaic_ccm / unpack_blc) halve resolution:
    // clear elementwise flag so ShapeExtra doesn't copy input shape, and
    // override global_size to reflect the 2× downscale in spatial dims.
    if (pat.ispType && (strcmp(pat.ispType, "isp.pyramid") == 0 || strcmp(pat.ispType, "isp.demosaic_ccm") == 0 ||
        strcmp(pat.ispType, "isp.unpack_blc") == 0 || strcmp(pat.ispType, "isp.unpack_demosaic") == 0)) {
        clearElementwise(ex);
        int outC = (pat.ispType && (strcmp(pat.ispType, "isp.pyramid") == 0 || strcmp(pat.ispType, "isp.unpack_blc") == 0)) ? 4 : 3;
        // Override global_size to [W/2, H/2, outC] → ShapeExtra output [1, outC, H/2, W/2]
        for (auto& a : ex->attr) {
            if (a->key == "global_size") {
                auto* lst = a->list.get();
                if (lst && lst->i.size() >= 3) {
                    lst->i[0] = mW / 2;  // gx
                    lst->i[1] = mH / 2;  // gy
                    lst->i[2] = outC;     // gz = channel count
                }
                break;
            }
        }
    }
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
    { std::string cs; for (int k = 0; k < (int)idx.size(); k++) { if (k) cs += ","; cs += std::to_string(idx[k]); } VLOG(2) << "[P1] CONSUME " << cs; }
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
    // Conv(5x5) with 300 weights, 4→3 (DebayerBlock learned debayer)
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 300),
    // Conv(depthwise, 3x3, 4ch) with 36 weights (real debayer)
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 36),
    // Conv(1x1, 12 weights, 4→3)→ReLU6 (Clip) with runtime-fused weights (DemosaicCcmBlock)
    ExactPattern({MNN::OpType_Convolution, MNN::OpType_ReLU6},
                 -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 12),
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
    // 2-op: BinaryOp(SUB)+ReLU6 -> isp.unpack_blc (UnpackBlc16Block: Cast->Sub->Clip
    // after RemoveInvalidCast strips Cast, MNN maps Clip to ReLU6)
    ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
                 -1, -1, "isp.unpack_blc", "isp.unpack_blc",
                 MNN::BinaryOpOperation_SUB),
    // 2-op: Reshape->ReLU6->BinaryOp (RawBlcBlock variant)
    ExactPattern({MNN::OpType_Reshape, MNN::OpType_ReLU6, MNN::OpType_BinaryOp},
                 -1, -1, "isp.unpack_blc", "isp.unpack_blc", nullptr),
    // 1-op: Sub with 1-elem const (RawBlcBlock) — checked via constElems
    // Note: must be BinaryOp(SUB), but matchExact only checks type. This is
    // (Exact tables handle all ISP pattern matching.)
    // kExactNormalizeBlock handles Div(1-elem) with a separate table.
};

static const ExactPattern kExactPyramid[] = {
    // CfaBlock: Conv(2x2, stride=2, 16 weights, 1→4) → pyramid (Bayer quad unpack)
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.pyramid", "isp.pyramid", nullptr, 16),
    // CfaBlock variant: Conv(2x2, stride=2, 4 weights, 1→4) → pyramid
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.pyramid", "isp.pyramid", nullptr, 4),
};

static const ExactPattern kExactColorspace[] = {
    // CscBlock: Conv(1x1, 3→3, 9 weights) with 3-elem bias → isp.colorspace
    // Must run BEFORE kExactFcs (which also matches 9-weight Conv)
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.colorspace", "isp.colorspace", "colorspace", 9),
};

static const ExactPattern kExactNormalize[] = {
    // NormalizeBlock: Cast→RealDiv(1-elem const, FP division) → isp.fcs
    // NOTE: uses REALDIV (floating-point), not DIV (integer).
    ExactPattern({MNN::OpType_Cast, MNN::OpType_BinaryOp},
                 1, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_REALDIV),
    // 2-op entry (new lib pass1): Cast(DT_VARIANT→DT_FLOAT) survives because
    // INT16→DT_INT32 mapping makes the Cast valid. Chain = Cast→BinaryOp(REALDIV).
    // Profile-variant ctor: const checks apply to idx[0] (the Cast, 1 input) so
    // they must stay off; per-position chainBinOps pins idx[1] to REALDIV.
    // Mirrors kExactRawBlcBlock's 2-op Cast→SUB entry.
    ExactPattern({MNN::OpType_Cast, MNN::OpType_BinaryOp}, -1, -1,
                 "isp.fcs", "isp.fcs",
                 MNN::BinaryOpOperation_ADD, true, {}, {}, -1,
                 {{1, MNN::BinaryOpOperation_REALDIV}}, -1),
    // 1-op fallback: REALDIV with no Cast (old lib pass1, INT16→DT_FLOAT folded).
    ExactPattern({MNN::OpType_BinaryOp},
                 -1, -1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_REALDIV, true, {}),
};

static const ExactPattern kExactAe[] = {
    // WbGainsBlock: Mul(3-elem gains) → isp.ae
    ExactPattern({MNN::OpType_BinaryOp},
                 3, 1, "isp.ae", "isp.ae", MNN::BinaryOpOperation_MUL),
};

static const ExactPattern kExactAfFocus[] = {
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.af_focus", "isp.af_focus", nullptr, 9),
};

// CalibrationBlock: isp.calib_stats (21-op chain)
static const ExactPattern kExactCalibrationBlock[] = {
ExactPattern({MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_BinaryOp, MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_Squeeze, MNN::OpType_Reduction, MNN::OpType_Reshape, MNN::OpType_Reduction, MNN::OpType_Reshape, MNN::OpType_Reduction, MNN::OpType_Reshape, MNN::OpType_Reduction, MNN::OpType_Reshape, MNN::OpType_Concat}, -1, -1, "isp.calib_stats", "isp.calib_stats"),
};

// UnifiedStatsBlock: isp.ispc_stats (19-op chain)
static const ExactPattern kExactUnifiedStatsBlock[] = {
ExactPattern({MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_StridedSlice, MNN::OpType_StridedSlice, MNN::OpType_StridedSlice, MNN::OpType_Convolution, MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_Reduction, MNN::OpType_Squeeze, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_Concat, MNN::OpType_Squeeze}, -1, -1, "isp.ispc_stats", "isp.ispc_stats"),
};

// DisplayBlock: isp.display (16-op chain)
static const ExactPattern kExactDisplayBlock[] = {
ExactPattern({MNN::OpType_Permute, MNN::OpType_Padding, MNN::OpType_Shape, MNN::OpType_Rank, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_Unsqueeze, MNN::OpType_BinaryOp, MNN::OpType_Unsqueeze, MNN::OpType_StridedSlice, MNN::OpType_Squeeze, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_GatherV2, MNN::OpType_BinaryOp, MNN::OpType_Cast}, -1, -1, "isp.display", "isp.display"),
};

// FcsBlock: isp.fcs (13-op chain)
static const ExactPattern kExactFcsBlock[] = {
ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ConvolutionDepthwise, MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, 1, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_SUB),
};

// ToneBlock: isp.tone (12-op chain)
static const ExactPattern kExactToneBlock[] = {
ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, 1, 0, "isp.tone", "isp.tone", MNN::BinaryOpOperation_SUB),
};

// AlgoAwbBlock: isp.awb (11-op chain)
static const ExactPattern kExactAlgoAwbBlock[] = {
ExactPattern({MNN::OpType_Reduction, MNN::OpType_StridedSlice, MNN::OpType_StridedSlice, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_StridedSlice, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_Concat, MNN::OpType_Reshape}, -1, -1, "isp.awb", "isp.awb"),
};

// AlgoAeBlock: isp.ae (9-op chain)
static const ExactPattern kExactAlgoAeBlock[] = {
ExactPattern({MNN::OpType_StridedSlice, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6, MNN::OpType_BinaryOp, MNN::OpType_UnaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.ae", "isp.ae"),
};

// FocusBlock: isp.af_focus (8-op chain)
static const ExactPattern kExactFocusBlock[] = {
ExactPattern({MNN::OpType_Convolution, MNN::OpType_Convolution, MNN::OpType_UnaryOp, MNN::OpType_Convolution, MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_Reduction, MNN::OpType_Squeeze}, -1, -1, "isp.af_focus", "isp.af_focus", nullptr, 3),
};

// YuvSatBlock: isp.fcs (7-op chain)
static const ExactPattern kExactYuvSatBlock[] = {
ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, 1, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_SUB),
};

// AutoContrastBlock: isp.auto_contrast (6-op chain)
static const ExactPattern kExactAutoContrastBlock[] = {
    // Value-constrained FIRST: real AutoContrastBlock always has the four=4.0
    // const at chain pos 2 input 1 (scaledS = diffSq * 4.0). This shadows the
    // generic entry below for the real block while failing on scalar control
    // chains (algo_gamma Log→Div→Mul→Add→Mul→Sub and algo_cct n²→n³→… windows
    // have NO 4.0 const at that position).
    ExactPattern({MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp}, -1, -1, "isp.auto_contrast", "isp.auto_contrast", false, {ChainConstCheck(2, 1, {4.0f})}),
    // existing generic pattern (kept — duplicated; shadowed by the entry above)
    ExactPattern({MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp}, -1, -1, "isp.auto_contrast", "isp.auto_contrast"),
};

// ── no-fuse guards for scalar control chains ──
// algo_gamma / algo_cct are SCALAR camera-control chains ([1]-elem outputs) that
// MUST stay primitive. They get consumed by longest-match guards BEFORE any
// shorter generic image pattern (e.g. the 6-op isp.auto_contrast) can steal a
// sub-window and fuse it as a GPU shader running on scalar inputs (→ wrong
// shapes → heap corruption → SIGSEGV in Module output path).

// algo_gamma (8-op): Add(ag,eps)→Log→Div(ln10)→Mul(k1)→Add(base_gamma)→Mul(k2)→Sub→Clip
// NOTE: MNN folds Div(x,const) into Mul(x,1/const): op[2] const = 1/ln10 = 0.434294
static const ExactPattern kExactAlgoGammaChain[] = {
// variant A: Div folded → Mul with 1/ln10 = 0.434294 at pos2 input1
ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_UnaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
             -1, -1, "isp.noop_gamma", "isp.noop_gamma", /*noFuse=*/true,
             {ChainConstCheck(0, 1, {1e-6f}),      // eps
              ChainConstCheck(2, 1, {0.434294f}),  // 1/ln10 (Div folded to Mul)
              ChainConstCheck(3, 0, {0.2f}),       // k1
              ChainConstCheck(4, 0, {2.2f}),       // base_gamma
              ChainConstCheck(5, 0, {0.3f})}),     // k2
// variant B: Div kept → const ln10 = 2.302585 at pos2 input1
ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_UnaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
             -1, -1, "isp.noop_gamma", "isp.noop_gamma", /*noFuse=*/true,
             {ChainConstCheck(0, 1, {1e-6f}),      // eps
              ChainConstCheck(2, 1, {2.302585f}),  // ln10 (Div kept)
              ChainConstCheck(3, 0, {0.2f}),       // k1
              ChainConstCheck(4, 0, {2.2f}),       // base_gamma
              ChainConstCheck(5, 0, {0.3f})}),     // k2
};

// algo_cct (21-op): Slice×3→Add×3→Div×2→Sub×2→Add→Div→SQUARE→Mul×4→Add×3→Clip
// (n2 = SQUARE from Mul(nRatio,nRatio)); variant A: n2 is UnaryOp
// MNN topological order: ...n2, n3, term3, term2, sum1, term1, sum2, cct_raw, frame
// → term1 const c1 sits at chain pos 17 (not 16); c0 at pos 19.
static const ExactPattern kExactAlgoCctChain[] = {
ExactPattern({MNN::OpType_StridedSlice, MNN::OpType_StridedSlice, MNN::OpType_BinaryOp,
              MNN::OpType_StridedSlice, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
             -1, -1, "isp.noop_cct", "isp.noop_cct", /*noFuse=*/true,
             {ChainConstCheck(5, 1, {1e-6f}),        // eps in sum_eps
              ChainConstCheck(7, 1, {0.332f}),       // r_ref in r_shift
              ChainConstCheck(9, 1, {0.1858f}),      // b_ref in b_shift
              ChainConstCheck(14, 1, {-449.0f}),     // c3 in term3
              ChainConstCheck(15, 1, {3525.0f}),     // c2 in term2
              ChainConstCheck(17, 1, {-6823.3f}),    // c1 in term1 (pos 17)
              ChainConstCheck(19, 1, {5520.33f})}),  // c0 in cct_raw
};

// algo_cct variant B: n2 stays BinaryOp MUL (Mul(nRatio,nRatio) not folded)
static const ExactPattern kExactAlgoCctChainMul[] = {
ExactPattern({MNN::OpType_StridedSlice, MNN::OpType_StridedSlice, MNN::OpType_BinaryOp,
              MNN::OpType_StridedSlice, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp,
              MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
             -1, -1, "isp.noop_cct", "isp.noop_cct", /*noFuse=*/true,
             {ChainConstCheck(5, 1, {1e-6f}),        // eps in sum_eps
              ChainConstCheck(7, 1, {0.332f}),       // r_ref in r_shift
              ChainConstCheck(9, 1, {0.1858f}),      // b_ref in b_shift
              ChainConstCheck(14, 1, {-449.0f}),     // c3 in term3
              ChainConstCheck(15, 1, {3525.0f}),     // c2 in term2
              ChainConstCheck(17, 1, {-6823.3f}),    // c1 in term1 (pos 17)
              ChainConstCheck(19, 1, {5520.33f})}),  // c0 in cct_raw
};

// EeBlock atomic: isp.ee (1-op ConvDW with 27 weights = 3ch × 3×3 unsharp kernel)
static const ExactPattern kExactEeAtomicBlock[] = {
ExactPattern({MNN::OpType_ConvolutionDepthwise}, -1, -1, "isp.ee", "isp.ee", nullptr, 27),
};

// EeBlock: isp.ee (6-op chain)
static const ExactPattern kExactEeBlock[] = {
ExactPattern({MNN::OpType_ConvolutionDepthwise, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.ee", "isp.ee"),
};

// FastEeBlock: isp.ee (6-op chain)
static const ExactPattern kExactFastEeBlock[] = {
ExactPattern({MNN::OpType_ConvolutionDepthwise, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.ee", "isp.ee"),
};

// LdciBlock: isp.ldci (6-op with clip) — matched by TRY_EXACT_TABLE
static const ExactPattern kExactLdciBlock[] = {
ExactPattern({MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.ldci", "isp.ldci"),
};

// LdciBlockA: isp.ldci_a (4-op Pool→Sub→Mul→Add, no clip)
// Pins BinaryOp sub-types via chainBinOps so we don't match random 3-BinOp chains.
static const ExactPattern kExactLdciABlock[] = {
ExactPattern({MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp},
             -1, -1, "isp.ldci_a", "isp.ldci_a",
             MNN::BinaryOpOperation_ADD, true, {}, {}, -1,
             {{1, MNN::BinaryOpOperation_SUB},
              {2, MNN::BinaryOpOperation_MUL},
              {3, MNN::BinaryOpOperation_ADD}}),
};

// RefYuvSatBlock: isp.fcs (6-op chain)
static const ExactPattern kExactRefYuvSatBlock[] = {
ExactPattern({MNN::OpType_StridedSlice, MNN::OpType_StridedSlice, MNN::OpType_BinaryOp, MNN::OpType_Convolution, MNN::OpType_ReLU6, MNN::OpType_Concat}, -1, -1, "isp.fcs", "isp.fcs"),
};

// BilateralBlock: isp.bilateral (5-op chain)
static const ExactPattern kExactBilateralBlock[] = {
ExactPattern({MNN::OpType_ConvolutionDepthwise, MNN::OpType_ConvolutionDepthwise, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp}, -1, -1, "isp.bilateral", "isp.bilateral"),
};

// LocalContrastBlock: isp.fcs (5-op chain)
static const ExactPattern kExactLocalContrastBlock[] = {
ExactPattern({MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.fcs", "isp.fcs"),
};

// SaturationBlock: isp.fcs (5-op chain)
static const ExactPattern kExactSaturationBlock[] = {
ExactPattern({MNN::OpType_Reduction, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.fcs", "isp.fcs"),
};

// UnsharpBlock: isp.fcs (5-op chain)
static const ExactPattern kExactUnsharpBlock[] = {
ExactPattern({MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.fcs", "isp.fcs"),
};

// RefToneBlock: isp.tone (4-op chain)
static const ExactPattern kExactRefToneBlock[] = {
ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_ReLU, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, 1, 1, "isp.tone", "isp.tone", MNN::BinaryOpOperation_MUL),
};

// DemosaicCcmBlock: isp.demosaic_ccm (2-op chain)
static const ExactPattern kExactDemosaicCcmBlock[] = {
ExactPattern({MNN::OpType_Convolution, MNN::OpType_ReLU6}, -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 12),
};

// RefCcmBlock: isp.fcs (2-op chain)
static const ExactPattern kExactRefCcmBlock[] = {
ExactPattern({MNN::OpType_Convolution, MNN::OpType_BinaryOp}, -1, -1, "isp.fcs", "isp.fcs", nullptr, 9),
};

// VignettingBlock: isp.vignetting (2-op chain)
static const ExactPattern kExactVignettingBlock[] = {
ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, 1024, 1, "isp.vignetting", "isp.vignetting", MNN::BinaryOpOperation_MUL),
};

// BayerWbBlock: isp.fcs (1-op chain)
// Matches ANY BinaryOp(MUL) with 3 or 4 const elements — no scalar value check.
static const ExactPattern kExactBayerWbBlock[] = {
ExactPattern({MNN::OpType_BinaryOp}, 4, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_MUL),
ExactPattern({MNN::OpType_BinaryOp}, 3, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_MUL),
};

// BlcBlock: isp.fcs (1-op chain)
// Matches ANY BinaryOp(SUB) with 4 const elements — no scalar value check.
static const ExactPattern kExactBlcBlock[] = {
ExactPattern({MNN::OpType_BinaryOp}, 4, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_SUB),
};

// CcmBlock: isp.fcs (1-op chain)
// Matches ANY Convolution with 9 weights — no weight value check.
static const ExactPattern kExactCcmBlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.fcs", "isp.fcs", nullptr, 9),
};

// CfaBlock: isp.pyramid (1-op chain)
// Matches ANY Convolution with 16 weights — no weight value check.
static const ExactPattern kExactCfaBlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.pyramid", "isp.pyramid", nullptr, 16),
};

// ColorSpaceBlock: isp.colorspace (1-op chain)
static const ExactPattern kExactColorSpaceBlock[] = {
ExactPattern({MNN::OpType_MatMul}, -1, -1, "isp.colorspace", "isp.colorspace"),
};

// CscBlock: isp.colorspace (1-op chain)
// Matches ANY Convolution with 9 weights — no weight value check.
static const ExactPattern kExactCscBlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.colorspace", "isp.colorspace", nullptr, 9),
};

// DebayerBlock: isp.demosaic_debayer (1-op chain)
// Conv [3,4,5,5] = 300 weights (5x5 learned debayer 4ch→3ch)
static const ExactPattern kExactDebayerBlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.demosaic_debayer", "isp.demosaic_debayer", nullptr, 300),
};

// DemosaicBlock: isp.demosaic_a (1-op chain)
// Conv1x1 [3,4,1,1] = 12 weights, standalone bayer→RGB without clip/CCM.
// Heavy profile uses this variant (no ReLU6 following the Conv).
static const ExactPattern kExactDemosaicABlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.demosaic_a", "isp.demosaic_a", nullptr, 12),
};

// EdgeDemosaicBlock: isp.demosaic_edge (1-op chain)
// Conv [3,4,3,3] = 108 weights (3x3 kernel, edge-aware 4ch→3ch)
static const ExactPattern kExactEdgeDemosaicBlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.demosaic_edge", "isp.demosaic_edge", nullptr, 108),
};

// GammaBlock: isp.gamma (1-op chain)
// Matches ANY BinaryOp(POW) with 1-elem const exponent — no scalar value check.
// The algo_gamma guard (8-op noFuse, kExactAlgoGammaChain) runs first and
// protects scalar control chains, so this only matches standalone gamma curve ops.
static const ExactPattern kExactGammaBlock[] = {
ExactPattern({MNN::OpType_BinaryOp}, 1, 1, "isp.gamma", "isp.gamma", MNN::BinaryOpOperation_POW),
};

// HqLinearDemosaic: isp.demosaic_debayer (1-op chain)
// Conv [3,4,5,5] = 300 weights (5x5 linear demosaic 4ch→3ch)
static const ExactPattern kExactHqLinearDemosaic[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.demosaic_debayer", "isp.demosaic_debayer", nullptr, 300),
};

// LscBlock: isp.lsc (1-op chain)
static const ExactPattern kExactLscBlock[] = {
ExactPattern({MNN::OpType_BinaryOp}, 1024, 1, "isp.lsc", "isp.lsc", MNN::BinaryOpOperation_MUL),
};

// NormalizeBlock: isp.fcs (1-op chain)
// Matches ANY BinaryOp(MUL) with 1 const element — no scalar value check.
static const ExactPattern kExactNormalizeBlock[] = {
ExactPattern({MNN::OpType_BinaryOp}, 1, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_MUL),
};

// RawBlcBlock: isp.unpack_blc (2-op Cast→SUB chain for new lib)
// New lib preserves INT32 input → inserts Cast(DT_VARIANT→DT_FLOAT) before
// the SUB (input - offset). Old lib converted input to FLOAT → no Cast.
// 2-op entry (Cast→SUB) fires first; 1-op SUB is fallback for old lib.
static const ExactPattern kExactRawBlcBlock[] = {
    // 2-op: Cast→SUB (new lib pass1)
    ExactPattern({MNN::OpType_Cast, MNN::OpType_BinaryOp}, -1, -1,
                 "isp.unpack_blc", "isp.unpack_blc",
                 MNN::BinaryOpOperation_ADD, true, {}, {}, -1,
                 {{1, MNN::BinaryOpOperation_SUB}}, -1),
    // 1-op: SUB with 1 const element (old lib pass1, no Cast)
    ExactPattern({MNN::OpType_BinaryOp}, 1, 1, "isp.unpack_blc", "isp.unpack_blc", MNN::BinaryOpOperation_SUB),
};

// RefBayerWbBlock: isp.fcs (1-op chain)
// Matches ANY BinaryOp(MUL) with 4 const elements — no scalar value check.
static const ExactPattern kExactRefBayerWbBlock[] = {
ExactPattern({MNN::OpType_BinaryOp}, 4, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_MUL),
};

// RefCscBlock: isp.colorspace (1-op chain)
// Matches ANY Convolution with 9 weights — no weight value check.
static const ExactPattern kExactRefCscBlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.colorspace", "isp.colorspace", nullptr, 9),
};

// RefDebayerBlock: isp.demosaic_debayer (1-op chain)
static const ExactPattern kExactRefDebayerBlock[] = {
ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.demosaic_debayer", "isp.demosaic_debayer", nullptr, 300),
};

// WbGainsBlock: isp.awb (1-op chain)
// Matches ANY BinaryOp(MUL) with 3 const elements — no scalar value check.
static const ExactPattern kExactWbGainsBlock[] = {
ExactPattern({MNN::OpType_BinaryOp}, 3, 1, "isp.awb", "isp.awb", MNN::BinaryOpOperation_MUL),
};

// --- BEGIN_GENERATED_PRE_PASS1_TABLES ---
// ═══════════════════════════════════════════════════════════════
//  AUTO-GENERATED from clean PRE-PASS1 dumps (fusion disabled)
//  Profile-enabled blocks ONLY (PipelineProfile.kt buildBlocks)
//  Value-constrained: binOp/constElems/constVals/convW/convVals
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
//  AUTO-GENERATED from clean PRE-PASS1 dumps (fusion disabled)
//  Value-constrained: binOp/constElems/constVals/convW/convVals
// ═══════════════════════════════════════════════════════════════

static const ExactPattern kExactGeneratedDispatch[] = {
    // DisplayBlockYuv (chain=17)
        ExactPattern({MNN::OpType_Convolution, MNN::OpType_Permute, MNN::OpType_Padding, MNN::OpType_Shape, MNN::OpType_Rank, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_Unsqueeze, MNN::OpType_BinaryOp, MNN::OpType_Unsqueeze, MNN::OpType_StridedSlice, MNN::OpType_Squeeze, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_GatherV2, MNN::OpType_BinaryOp, MNN::OpType_Cast}, -1, -1, "isp.display", "isp.display", nullptr, 9, {1, 0, 1.402, 1, -0.344, -0.714, 1, 1.772, 0}),
    // DisplayBlock (chain=16)
        ExactPattern({MNN::OpType_Permute, MNN::OpType_Padding, MNN::OpType_Shape, MNN::OpType_Rank, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_Unsqueeze, MNN::OpType_BinaryOp, MNN::OpType_Unsqueeze, MNN::OpType_StridedSlice, MNN::OpType_Squeeze, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_GatherV2, MNN::OpType_BinaryOp, MNN::OpType_Cast}, -1, -1, "isp.display", "isp.display"),
    // BadPixelBlock (chain=10)
        ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6, MNN::OpType_BinaryOp, MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_ReLU6, MNN::OpType_BinaryOp}, 1, 0, "isp.dpc", "isp.dpc", MNN::BinaryOpOperation_SUB, nullptr, -1, {}, {0}),
    // YuvSatBlock (chain=7)
        ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, 1, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_SUB, nullptr, -1, {}, {0.5}),
    // AutoContrastBlock (chain=6)
        ExactPattern({MNN::OpType_UnaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp}, -1, -1, "isp.auto_contrast", "isp.auto_contrast"),
    // ChromaticAberration (chain=6)
        ExactPattern({MNN::OpType_StridedSlice, MNN::OpType_GridSample, MNN::OpType_StridedSlice, MNN::OpType_StridedSlice, MNN::OpType_GridSample, MNN::OpType_Concat}, -1, -1, "isp.warp", "isp.warp"),
    // BilateralBlock (chain=5)
        ExactPattern({MNN::OpType_ConvolutionDepthwise, MNN::OpType_ConvolutionDepthwise, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp}, -1, -1, "isp.bilateral", "isp.bilateral", nullptr, 15, {0.0625, 0.25, 0.375, 0.25, 0.0625, 0.0625, 0.25, 0.375, 0.25, 0.0625, 0.0625, 0.25, 0.375, 0.25, 0.0625}),
    // LocalContrastBlock (chain=5)
        ExactPattern({MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.fcs", "isp.fcs"),
    // SaturationBlock (chain=5)
        ExactPattern({MNN::OpType_Reduction, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.fcs", "isp.fcs"),
    // UnsharpBlock (chain=5)
        ExactPattern({MNN::OpType_Pooling, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, -1, -1, "isp.fcs", "isp.fcs"),
    // VignettingBlock (chain=2)
        ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_ReLU6}, 1024, 1, "isp.vignetting", "isp.vignetting", MNN::BinaryOpOperation_MUL, nullptr, -1, {}),
    // BayerWbBlock (chain=1)
        ExactPattern({MNN::OpType_BinaryOp}, 4, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_MUL, nullptr, -1, {}, {1, 1, 1, 1}),
    // BlcBlock (chain=1)
        ExactPattern({MNN::OpType_BinaryOp}, 4, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_SUB, nullptr, -1, {}, {0, 0, 0, 0}),
    // CcmBlock (chain=1)
        ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.fcs", "isp.fcs", nullptr, 9, {1, 1, 1, 1, 1, 1, 1, 1, 1}),
    // CfaBlock (chain=1)
        ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.pyramid", "isp.pyramid", nullptr, 16, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
    // CropBlock (chain=1)
        ExactPattern({MNN::OpType_StridedSlice}, -1, -1, "isp.fcs", "isp.fcs"),
    // CscBlock (chain=1)
        ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.colorspace", "isp.colorspace", nullptr, 9, {0.299, 0.587, 0.114, -0.169, -0.331, 0.5, 0.5, -0.419, -0.081}),
    // CscBlockYuv2Rgb (chain=1)
        ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.colorspace", "isp.colorspace", nullptr, 9, {1, 0, 1.402, 1, -0.344, -0.714, 1, 1.772, 0}),
    // DebayerBlock (chain=1)
        ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 300),
    // DownscaleBlock (chain=1)
        ExactPattern({MNN::OpType_Interp}, -1, -1, "isp.interp", "isp.interp"),
    // EdgeDemosaicBlock (chain=1)
        ExactPattern({MNN::OpType_Convolution}, -1, -1, "isp.demosaic_ccm", "isp.demosaic_ccm", nullptr, 108),
    // GammaBlock (chain=1)
        ExactPattern({MNN::OpType_BinaryOp}, 1, 1, "isp.gamma", "isp.gamma", MNN::BinaryOpOperation_POW, nullptr, -1, {}, {1}),
    // LscBlock (chain=1)
        ExactPattern({MNN::OpType_BinaryOp}, 1024, 1, "isp.lsc", "isp.lsc", MNN::BinaryOpOperation_MUL, nullptr, -1, {}),
    // NormalizeBlock (chain=1)
        ExactPattern({MNN::OpType_BinaryOp}, 1, 1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_MUL, nullptr, -1, {}, {1}),
    // RawBlcBlock (chain=1)
        ExactPattern({MNN::OpType_BinaryOp}, 1, 1, "isp.unpack_blc", "isp.unpack_blc", MNN::BinaryOpOperation_SUB, nullptr, -1, {}, {0}),
};

// Total tables: 25

// --- END_GENERATED_PRE_PASS1_TABLES ---

// ═══════════════════════════════════════════════════════════════════
// PROFILE-VARIANT tables: profile pass0 graphs feed block constants as
// graph Inputs (runtime tensors), not Const ops — so the generated
// (value-constrained) patterns can never match those blocks. These variants
// relax the const check and disambiguate by structural provenance:
//   inputTrace         — required producer op type (through ConvertTensor)
//   inputMustBeInput   — input fed directly by a graph Input
//   nextOpType         — required type of the next non-CT/Const/Input op
//   chainBinOps        — per-chain-position BinaryOp sub-type
// Sorted longest-first; RawBlcBlock MUST precede BlcBlock (both 1-op SUB
// with an Input at input[1] — RawBlc wins via the StridedSlice trace).
//
// Derived from the MANUAL ALIGNMENT report (align_profile_blocks.py):
//   .agent/data/tmp/alignment_report.txt
static const ExactPattern kExactProfileVariants[] = {
    // BayerWbBlock (4-op, gains-reorder wiring): profile adds
    // Reshape→Conv(gains reorder)→Reshape→MUL (standalone is bare MUL 4e).
    ExactPattern({MNN::OpType_Reshape, MNN::OpType_Convolution, MNN::OpType_Reshape, MNN::OpType_BinaryOp},
                 -1, -1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_ADD, true, {}, {}, -1,
                 {{3, MNN::BinaryOpOperation_MUL}}),
    // VignettingBlock (2-op): MUL(radius map Input)→ReLU6 (standalone const 1024e).
    ExactPattern({MNN::OpType_BinaryOp, MNN::OpType_ReLU6},
                 -1, -1, "isp.vignetting", "isp.vignetting", MNN::BinaryOpOperation_MUL, true, {}, {1}),
    // RawBlcBlock (1-op): SUB with input1 = graph Input (tile const) whose next
    // op is REALDIV (Normalize). nextBinOp disambiguates from BlcBlock (next MUL)
    // without tracing through the already-fused crop Extra.
    ExactPattern({MNN::OpType_BinaryOp},
                 -1, -1, "isp.unpack_blc", "isp.unpack_blc", MNN::BinaryOpOperation_SUB, true, {}, {1}, -1, {},
                 MNN::BinaryOpOperation_REALDIV),
    // BlcBlock (1-op): SUB with input1 = graph Input (blc offset) whose next op
    // is MUL (Lsc or BayerWb).
    ExactPattern({MNN::OpType_BinaryOp},
                 -1, -1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_SUB, true, {}, {1}, -1, {},
                 MNN::BinaryOpOperation_MUL),
    // BlcBlock variant (REF/INF, no Lsc): next op is the BayerWb gains-reorder
    // Reshape (Reshape→Conv→Reshape→MUL), not a MUL.
    ExactPattern({MNN::OpType_BinaryOp},
                 -1, -1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_SUB, true, {}, {1},
                 MNN::OpType_Reshape),
    // NormalizeBlock (1-op): profile emits REALDIV (x / sensor_max);
    // standalone emits MUL(1/max) which the generated pattern handles.
    ExactPattern({MNN::OpType_BinaryOp},
                 -1, -1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_REALDIV, true, {}),
    // GammaBlock (1-op): POW with gamma as graph Input (standalone const {2.2}).
    ExactPattern({MNN::OpType_BinaryOp},
                 -1, -1, "isp.gamma", "isp.gamma", MNN::BinaryOpOperation_POW, true, {}),
    // LscBlock (1-op): MUL(gain grid Input). In profiles the gain grid is a
    // graph Input (standalone bakes a 1024e const which the generated LscBlock
    // pattern handles). Fused as isp.fcs: per-pixel gain ops are opset-
    // equivalent to the fcs family (user directive: leave as-is).
    ExactPattern({MNN::OpType_BinaryOp},
                 -1, -1, "isp.fcs", "isp.fcs", MNN::BinaryOpOperation_MUL, true, {}, {1}),
    // CcmBlock (1-op): dynamic (w=0) convolution — matrix is a graph Input
    // (standalone bakes 9 weights which the generated CcmBlock pattern handles).
    ExactPattern({MNN::OpType_Convolution},
                 -1, -1, "isp.fcs", "isp.fcs", "fcs", 0),
};

static bool tryExactFirst(std::vector<std::unique_ptr<MNN::OpT>>& ops,
                          int& i, int mW, int mH) {
    bool matched = false;
#define TRY_EXACT_TABLE(tbl) do { \
        for (auto& pat : tbl) { \
            if (matchExact(ops, i, pat)) { \
                matched = true; \
                if (pat.noFuse) { \
                    /* Guard: consume chain (advance scan past it) but keep ops primitive. */ \
                    std::vector<int> gidx; std::vector<MNN::OpType> gtypes; \
                    collectChain(ops, i, (int)pat.opTypes.size(), gidx, gtypes); \
                    int guardEnd = gidx.empty() ? i : gidx.back(); \
                    VLOG(2) << "[P1] GUARD(noFuse) " << pat.ispType \
                            << " at " << i << " chain=" << (int)gidx.size() \
                            << " end=" << guardEnd; \
                    i = guardEnd; \
                    return true; \
                } \
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




    // ── GLOBAL LONGEST-FIRST dispatch (all tables merged & sorted) ──
    // Same length → generated (value-constrained) before existing generic.
    // Only tables for PROFILE-ENABLED blocks are dispatched (PipelineProfile.kt
    // buildBlocks). Disabled tables (definitions retained for reference):
    //   AlgoCct*, Calibration, UnifiedStats, FcsBlock, ToneBlock, AlgoAwb,
    //   AlgoAe, AlgoGamma, Focus, Ee*, FastEe, RefYuvSat,
    //   RefTone, Unpack, DemosaicCcm, RefCcm, Demosaic, ColorSpace, HqLinear,
    //   RefBayerWb, RefCsc, RefDebayer, WbGains, Fcs(1op), Pyramid, Colorspace,
    //   Ae, AfFocus.
    TRY_EXACT_TABLE(kExactGeneratedDispatch);          // generated from pre-pass1 dumps (profile blocks only)
    TRY_EXACT_TABLE(kExactProfileVariants);          // profile Input-const variants (structural)
    TRY_EXACT_TABLE(kExactDisplayBlock);          // 16-op (gen)
    TRY_EXACT_TABLE(kExactYuvSatBlock);          // 7-op (gen)
    TRY_EXACT_TABLE(kExactAutoContrastBlock);          // 6-op (gen)
    TRY_EXACT_TABLE(kExactDisplay);          // 6-op (existing)
    TRY_EXACT_TABLE(kExactLdciBlock);          // 6-op (Rust LdciBlock with clip)
    TRY_EXACT_TABLE(kExactLdciABlock);         // 4-op (Rust LdciBlockA, no clip)
    TRY_EXACT_TABLE(kExactBilateralBlock);          // 5-op (gen)
    TRY_EXACT_TABLE(kExactLocalContrastBlock);          // 5-op (gen)
    TRY_EXACT_TABLE(kExactSaturationBlock);          // 5-op (gen)
    TRY_EXACT_TABLE(kExactUnsharpBlock);          // 5-op (gen)
    TRY_EXACT_TABLE(kExactVignettingBlock);          // 2-op (gen)
    TRY_EXACT_TABLE(kExactNormalize);          // 2-op (existing)
    TRY_EXACT_TABLE(kExactBayerWbBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactBlcBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactCcmBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactCfaBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactCscBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactDebayerBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactEdgeDemosaicBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactGammaBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactLscBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactNormalizeBlock);          // 1-op (gen)
    TRY_EXACT_TABLE(kExactDemosaicCcmBlock);          // 2-op (Conv+ReLU6, 12 weights)
    TRY_EXACT_TABLE(kExactDemosaicABlock);             // 1-op (Conv 1x1, 12 weights, standalone demosaic)
    TRY_EXACT_TABLE(kExactEeAtomicBlock);             // 1-op (ConvDW, 27 weights)
    TRY_EXACT_TABLE(kExactRawBlcBlock);          // 1-op (gen)

    // ── FAIL CASE: no exact pattern matched at i. ──
    // The op is left untouched → stays in the primitive opset (surfaced by the
    // JSON opset gate, not by converter logging).
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
    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;

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

            // ── Exact-pattern pre-dispatch ──
            if (tryExactFirst(ops, i, mW, mH)) { changed = true; continue; }
            // TODO: enable disabled exact tables after verification:
            //   kExactPyramid, kExactRefYuvSatBlock, kExactRefBayerWbBlock
            //   (definitions exist in tables above, not yet dispatched)

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

    // TODO: add exact tables for removed try* patterns:
    //   tryDisplay → needs kExactDisplayPowBlock (Pow+Cast+Conv chain)
    //   tryPyramid → kExactPyramid exists but not dispatched
    //   tryRustDisplay → single Mul(identity) → isp.display
    //   tryVignetting → kExactVignettingBlock already dispatched
    //   tryAutoContrast → kExactAutoContrastBlock already dispatched
    //   tryDpc → 5-op Pool→Sub→Mul→Add→Clip
    //   tryLsc → kExactLscBlock already dispatched
    //   tryGamma → kExactGammaBlock already dispatched
    //   tryUnsharp → kExactUnsharpBlock already dispatched
    //   trySaturation → kExactSaturationBlock already dispatched
    //   tryBadPixel → always returned false (dead)
    //   tryBayerWb → kExactBayerWbBlock already dispatched
    //   tryBilateral → kExactBilateralBlock already dispatched
    //   tryYuvSat → kExactYuvSatBlock already dispatched
    //   tryBLC → kExactBlcBlock already dispatched
    //   tryNormalize → kExactNormalize already dispatched
    //   tryYuvSat7 → kExactRefYuvSatBlock exists but not dispatched
    //   tryBayerWbReshape → kExactRefBayerWbBlock exists but not dispatched
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
        // ═══ PASS2 FUSION DISABLED (session goal) ═══
        // Merging isp.* Extras is deferred: Pass1 pattern matching is the
        // focus of validation. Pass2 renamed isp.display→isp.fcs_display
        // (R8 rule), breaking the 1:1 profile gate.
        (void)net;
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
        // ── Format-based ISP enable gate ──
        // Fusion runs ONLY for MNN→MNN pass2; ONNX→MNN pass1 always stays
        // primitive (Conv, BinaryOp, etc.) so all backends handle it. With
        // fusion, ISP Extra ops embed SPIR-V instead.
        // modelConfig is set by cli.cpp based on input format, or by the
        // JNI bridge (mnn_convert_api.cpp) for MNN→MNN.
        {
            bool enableIsp = false;
            auto* cfg = Global<modelConfig>::Get();
            if (cfg != nullptr && cfg->model == modelConfig::MNN) {
                enableIsp = true;
            }
            if (!enableIsp) {
                VLOG(1) << "[IspFusion] input is not MNN — skipping ISP fusion (pass1 simple convert)";
                return true;
            }
        }

        VLOG(1) << "[IspFusion] === Pre-pass: Remove Identity ops ===";
        RemoveIdentityOps().onExecute(net);

        VLOG(1) << "[IspFusion] === Pass 1: Standard → ISP Extra ops ===";
        Pass1_ToExtra::instance()->onExecute(net);

        VLOG(1) << "[IspFusion] === Pass 2: MACRO-FUSION — Extra chain → fused Extra ===";
        Pass2_FuseExtra::instance()->onExecute(net);

        VLOG(1) << "[IspFusion] Complete: " << net->oplists.size() << " ops";
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
    
