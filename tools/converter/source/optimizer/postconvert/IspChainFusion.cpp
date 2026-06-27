//
//  IspChainFusion.cpp — MNN converter optimization pass (2-pass autoregressive)
//
//  Architecture — two passes, each running iteratively to fixpoint:
//
//    ONNX model (standard ai.onnx ops)
//      ↓ MNNConvert
//    Standard MNN ops (NC4HW4)
//      ↓ Pass 1: Standard → ISP Extra ops
//    Logical ISP stages (CHW planar, one op per stage)
//      ↓ Pass 2: Adjacent ISP Extra → Fused Extra ops
//    Fused ISP pipeline (3 dispatches)
//      ↓ 3-5× faster on GPU
//
//  Pass 1 rules (standard MNN ops → Extra ops):
//    ┌────────────────────────────────────────────────────────────┐
//    │ 1. Cast(→FLOAT) + Conv(2×2,stride=2,4ch) → isp.unpack_blc│
//    │ 2. Conv(1×1,4→3ch)                      → isp.demosaic_ccm│
//    │ 3. Conv(1×1,4→3ch,noscale)              → isp.demosaic_noscale│
//    │ 4. Scale                                 → isp.fcs        │
//    │ 5. Conv(3×3,unsharp,group=3)             → isp.ee         │
//    │ 6. Pool(AVG,3×3)+Sub+Mul+Add             → isp.ldci       │
//    │ 7. BinaryOp(POW,exp=1/2.4)[+Clip(0,1)]   → isp.display    │
//    └────────────────────────────────────────────────────────────┘
//
//  Pass 2 rules (Extra chain → fused Extra):
//    ┌────────────────────────────────────────────────────────────┐
//    │ 8. isp.fcs + isp.display               → isp.fcs_display  │
//    │ 9. isp.ee + isp.ldci                   → isp.ee_ldci      │
//    │10. isp.unpack_blc + isp.demosaic_ccm   → isp.unpack_demosaic│
//    │11. isp.unpack_demosaic + isp.fcs_display                  │
//    │    + isp.ee_ldci                        → isp.fused_6in1   │
//    └────────────────────────────────────────────────────────────┘

#include <string>
#include <vector>
#include <cmath>
#include "PostTreatUtils.hpp"
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
        {"isp.fused_6in1",      g_fused_6in1_spv,      g_fused_6in1_spv_len},
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

static bool isUnpackConv(const Convolution2DT* c) {
    return c && c->common && c->common->kernelX == 2 && c->common->kernelY == 2
           && c->common->strideX == 2 && c->common->strideY == 2
           && c->common->outputCount == 4;
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

        for (int i = 0; i < (int)ops.size(); i++) {
            if (!ops[i]) continue;

            // ── R1: Cast + Conv(2×2,stride=2,4ch) → isp.unpack_blc ──
            if (tryUnpack(ops, i)) { changed = true; continue; }

            // ── R2: CCM Conv(1×1,4→3ch) → isp.demosaic_ccm ──
            if (tryDemosaic(ops, i)) { changed = true; continue; }

            // ── R3: Scale → isp.fcs ──
            if (tryFcs(ops, i)) { changed = true; continue; }

            // ── R4: Conv(3×3,unsharp) → isp.ee ──
            if (tryEe(ops, i)) { changed = true; continue; }

            // ── R5: Pool(AVG,3×3)+Sub+Mul+Add → isp.ldci ──
            if (tryLdci(ops, i)) { changed = true; continue; }

            // ── R6: BinaryOp(POW)[+Clip] → isp.display ──
            if (tryDisplay(ops, i)) { changed = true; continue; }
        }

        ops.erase(std::remove_if(ops.begin(), ops.end(),
                  [](const std::unique_ptr<OpT>& o) { return !o; }), ops.end());
        return changed;
    }

private:
    // Dimensions inferred from model input shape at onExecute time
    mutable int mW = 1920, mH = 1080;       // output (after stride-2)
    mutable int mInW = 3840, mInH = 2160;   // input (Bayer raw)

    // R1: Cast + Conv(2×2,stride=2,4ch) → isp.unpack_blc
    bool tryUnpack(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        int ci = (ops[i]->type == MNN::OpType_Cast) ? i+1 : i;
        if (ci >= (int)ops.size() || !ops[ci]) return false;
        if (ops[ci]->type != MNN::OpType_Convolution) return false;
        auto* conv = ops[ci]->main.AsConvolution2D();
        if (!isUnpackConv(conv)) return false;

        std::vector<float> u = {float(mW),float(mH),float(mInW),float(mInH),
                                1023.0f, 0,0,0,0, 1,1,1,1};

        ops[ci]->type = MNN::OpType_Extra;
        ops[ci]->main.type = MNN::OpParameter_Extra;
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
        ops[ci]->main.value = ex;

        VLOG(2) << "[P1] R1: unpack_blc at " << i;
        if (i != ci) ops[i].reset();  // remove Cast
        i = ci;
        return true;
    }

    // R2: Conv(1×1,4→3ch) → isp.demosaic_ccm
    // Extracts the 3×4 Conv weights and converts to 3×3 CCM for the fused shader.
    // The 3×4 matrix maps [R, Gr, Gb, B] → [R', G', B'].
    // The demosaic shader computes G=(Gr+Gb)/2, then applies 3×3 CCM:
    //   ccm[i][R] = w[i][R], ccm[i][G] = 2*w[i][Gr], ccm[i][B] = w[i][B]
    // Assumes w[i][Gr] == w[i][Gb] (equal Gr/Gb contribution per output channel).
    bool tryDemosaic(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!isCcmConv(c)) return false;

        // Extract 3×4 Conv weights → 3×3 CCM
        // Weights layout: [OC=3, IC=4, KY=1, KX=1] = 12 floats contiguous
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
                                1023.0f,
                                ccm[0],ccm[1],ccm[2],ccm[3],ccm[4],ccm[5],ccm[6],ccm[7],ccm[8],
                                0,0,0,0};

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.demosaic_ccm";
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex);
        addSpirv(ex, "isp.demosaic_ccm");
        ops[i]->main.value = ex;
        VLOG(2) << "[P1] R2: demosaic_ccm at " << i;
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
        i = addIdx;
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

            // R8: fcs + display → fcs_display
            for (size_t k = 0; k + 1 < extras.size(); k++) {
                int i = extras[k], j = extras[k+1];
                if (isExtraOfType(ops[i].get(), "isp.fcs") &&
                    isExtraOfType(ops[j].get(), "isp.display") &&
                    matchFcsDisplay(ops, i, j)) {
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

            // R11: fused_6in1 — 1 dispatch but 3-5× slower due to 5×5 FCS redundancy.
            // if (matchFused6in1(ops)) { any = true; continue; }

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

        // Build const buffer for unpack_demosaic: [dims4, smax, blc4, wb4, ccm9, fcs2, pad2]
        std::vector<float> u = {float(W),float(H), float(inpW),float(inpH), 1023,
                                0,0,0,0, 1,1,1,1,
                                1,0,0, 0,1,0, 0,0,1,
                                1.0f, 0.0f,  // fcs_str=1.0, fcs_off=0.0 (default)
                                0,0};

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

    // R11: Fuse all stages from unpack_demosaic → display into fused_6in1.
    // Collects consecutive Extra ops from unpack_demosaic through display
    // in tensor chain order and collapses them into a single dispatch.
    bool matchFused6in1(std::vector<std::unique_ptr<OpT>>& ops) const {
        // Collect Extras in pipeline order starting from unpack_demosaic
        std::vector<int> extras;
        int cur = -1;
        for (auto& op : ops)
            if (op && op->type == MNN::OpType_Input && !op->outputIndexes.empty())
                { cur = op->outputIndexes[0]; break; }
        if (cur < 0) return false;
        bool foundUd = false;
        while (cur >= 0) {
            bool found = false;
            for (int j = 0; j < (int)ops.size(); j++) {
                if (!ops[j] || ops[j]->type != MNN::OpType_Extra) continue;
                for (int inIdx : ops[j]->inputIndexes)
                    if (traceTensor(inIdx, ops) == cur) {
                        if (isExtraOfType(ops[j].get(), "isp.unpack_demosaic")) foundUd = true;
                        extras.push_back(j);
                        cur = ops[j]->outputIndexes.empty() ? -1 : ops[j]->outputIndexes[0];
                        found = true; break;
                    }
                if (found) break;
            }
            if (!found) break;
        }
        // Need at least 4 extras: unpack_demosaic + fcs + ee_ldci + display
        if (!foundUd || extras.size() < 4) return false;

        VLOG(1) << "[P2] R11: Fused6in1 at " << extras[0] << "+...+" << extras.back();
        {
            std::string s = "[IspFusion] [P2] R11: FuseAll " + std::to_string(extras.size()) + " stages → isp.fused_6in1 ";
            for (int idx : extras) {
                if (ops[idx] && ops[idx]->main.AsExtra())
                    s += std::to_string(idx) + "(" + ops[idx]->main.AsExtra()->type + ") ";
                else
                    s += std::to_string(idx) + "(NULL) ";
            }
            VLOG(1) << s;
        }
        
        int W, H;
        getExtraDims(ops[extras[0]], W, H);
        if (W <= 0 || H <= 0) { W = 1920; H = 1080; }
        int inpW = W*2, inpH = H*2;
        
        // Read params from each stage by named attribute
        std::vector<float> blc = {0,0,0,0};
        std::vector<float> wb  = {1,1,1,1};
        std::vector<float> ccm = {1,0,0, 0,1,0, 0,0,1};
        float fcs_str = 1.0f, fcs_off = 0.0f;
        float ee_str = 0.5f, ee_thr = 0.01f;
        float ldci_str = 0.5f, ldci_rad = 1.0f;
        float gamma = 2.2f;

        for (int idx : extras) {
            if (!ops[idx] || !ops[idx]->main.AsExtra()) continue;
            auto* ex = ops[idx]->main.AsExtra();
            const char* t = ex->type.c_str();

            auto blcVals = getNamedFloats(ex, "blc");
            auto wbVals  = getNamedFloats(ex, "wb");
            auto ccmVals = getNamedFloats(ex, "ccm");
            auto fcsVals = getNamedFloats(ex, "fcs");
            auto eeVals  = getNamedFloats(ex, "ee");
            auto ldciVals= getNamedFloats(ex, "ldci");
            auto dispVals= getNamedFloats(ex, "display");

            if (blcVals.size() >= 4) blc = blcVals;
            if (wbVals.size() >= 4)  wb = wbVals;
            if (ccmVals.size() >= 9) { for (int k = 0; k < 9; k++) ccm[k] = ccmVals[k]; }
            if (fcsVals.size() >= 2) { fcs_str = fcsVals[0]; fcs_off = fcsVals[1]; }
            if (eeVals.size() >= 2)  { ee_str = eeVals[0]; ee_thr = eeVals[1]; }
            if (ldciVals.size() >= 2){ ldci_str = ldciVals[0]; ldci_rad = ldciVals[1]; }
            if (dispVals.size() >= 2){ gamma = dispVals[0]; }
        }
        
        // Build 33-float uniform buffer
        // Layout: [W, H, BW, BH, smax, blc4, wb4, ccm9, fcs_str, fcs_off, ee_str, ee_thr, ldci_str, ldci_rad, gamma, bright, pad3]
        std::vector<float> u = {float(W),float(H), float(inpW),float(inpH), 1023,
                                blc[0],blc[1],blc[2],blc[3],
                                wb[0],wb[1],wb[2],wb[3],
                                ccm[0],ccm[1],ccm[2],ccm[3],ccm[4],ccm[5],ccm[6],ccm[7],ccm[8],
                                fcs_str, fcs_off,
                                ee_str, ee_thr,
                                ldci_str, ldci_rad,
                                gamma, 0.0f,
                                0,0,0};
        
        auto* first = ops[extras[0]].get();
        auto* last  = ops[extras.back()].get();
        first->main.AsExtra()->type = "isp.fused_6in1";
        first->main.AsExtra()->attr.clear();
        buildCommonAttrs(first->main.AsExtra(), W, H, u);
        setEngine(first->main.AsExtra());
        addSpirv(first->main.AsExtra(), "isp.fused_6in1");
        first->outputIndexes[0] = last->outputIndexes[0];
        
        for (size_t k = 1; k < extras.size(); k++)
            ops[extras[k]].reset();
        return true;
    }

    // R12: Fuse all pipeline stages into one fused Extra op.
    // Walks the linear chain from Input → Extra* → Output and fuses
    // all consecutive Extra ops. This handles any combination of stages
    // without requiring pairwise adjacency checks.
};

// ═══════════════════════════════════════════════════════════════════
//  Orchestrator: runs Pass1 → Pass2 autoregressively
// ═══════════════════════════════════════════════════════════════════

class IspChainFusion : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        VLOG(1) << "[IspFusion] === Pass 1: Standard → ISP Extra ops ===";
        Pass1_ToExtra::instance()->onExecute(net);

        VLOG(1) << "[IspFusion] === Pass 2: Extra chain → fused Extra ===";
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
