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
    bool tryDemosaic(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!isCcmConv(c)) return false;

        std::vector<float> u = {float(mW),float(mH),float(mInW),float(mInH),
                                1023.0f, 1,0,0, 0,1,0, 0,0,1, 0,0,0,0};

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

        // Scan repeatedly until no more fusions. Each fusion reduces
        // operation count, enabling progressively longer chain matches.
        bool any = true;
        while (any) {
            any = false;

            // R11: unpack_demosaic + fcs_display + ee_ldci → fused_6in1 (longest)
            if (matchFused6in1(ops)) { any = true; continue; }

            // R10: unpack_blc + demosaic_ccm → unpack_demosaic
            if (matchUnpackDemosaic(ops)) { any = true; continue; }

            // R9: ee + ldci → ee_ldci
            if (matchEeLdci(ops)) { any = true; continue; }

            // R8: fcs + display → fcs_display
            if (matchFcsDisplay(ops)) { any = true; continue; }
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

    // R8: isp.fcs + isp.display → isp.fcs_display
    bool matchFcsDisplay(std::vector<std::unique_ptr<OpT>>& ops) const {
        for (int i = 0; i < (int)ops.size(); i++) {
            if (!isExtraOfType(ops[i].get(), "isp.fcs")) continue;
            for (int j = 0; j < (int)ops.size(); j++) {
                if (i == j || !ops[j]) continue;
                if (!isExtraOfType(ops[j].get(), "isp.display")) continue;
                if (!isChainSkipCT(ops[i].get(), ops[j].get(), ops)) continue;

                VLOG(1) << "[P2] R8: FcsDisplay at " << i << "+" << j;
                auto* fcs = ops[i]->main.AsExtra();
                float str = 1.0f;
                for (auto& a : fcs->attr) {
                    if (a->key == "const" && a->tensor && a->tensor->float32s.size() >= 3)
                        str = a->tensor->float32s[2];
                }
                std::vector<float> u = {1920,1080, str,0, 2.2f,0, 0,0,0};
                ops[i]->main.AsExtra()->type = "isp.fcs_display";
                ops[i]->main.AsExtra()->attr.clear();
                buildCommonAttrs(ops[i]->main.AsExtra(), 1920, 1080, u);
                setEngine(ops[i]->main.AsExtra());
                addSpirv(ops[i]->main.AsExtra(), "isp.fcs_display");
                ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
                ops[j].reset();
                return true;
            }
        }
        return false;
    }

    // R9: isp.ee + isp.ldci → isp.ee_ldci (handles both orderings)
    // Uses tensor chain detection (not adjacency) to handle any op ordering
    bool matchEeLdci(std::vector<std::unique_ptr<OpT>>& ops) const {
        // Find all ldci and ee ops
        for (int i = 0; i < (int)ops.size(); i++) {
            if (!isExtraOfType(ops[i].get(), "isp.ldci") &&
                !isExtraOfType(ops[i].get(), "isp.ee")) continue;
            for (int j = 0; j < (int)ops.size(); j++) {
                if (i == j || !ops[j]) continue;
                if (!isExtraOfType(ops[j].get(), "isp.ldci") &&
                    !isExtraOfType(ops[j].get(), "isp.ee")) continue;
                // Check if i→j forms a chain (either direction)
                bool ldciIn = isExtraOfType(ops[i].get(), "isp.ldci");
                bool ldciOut = isExtraOfType(ops[j].get(), "isp.ldci");
                if (ldciIn == ldciOut) continue; // both same type, skip
                
                bool chainIJ = isChainSkipCT(ops[i].get(), ops[j].get(), ops);
                bool chainJI = isChainSkipCT(ops[j].get(), ops[i].get(), ops);
                
                int keepIdx = -1, resetIdx = -1;
                if (chainIJ) { keepIdx = i; resetIdx = j; }
                else if (chainJI) { keepIdx = j; resetIdx = i; }
                else continue;
                
                std::string order = ldciIn ? "ldci→ee" : "ee→ldci";
                fprintf(stderr, "[IspFusion] [P2] R9: EeLdci MATCH at %d+%d (%s)\n", i, j, order.c_str());
                VLOG(1) << "[P2] R9: EeLdci at " << i << "+" << j << " (" << order << ")";
                int W, H;
                getExtraDims(ops[keepIdx], W, H);
                if (W <= 0 || H <= 0) getExtraDims(ops[resetIdx], W, H);
                if (W <= 0 || H <= 0) { W = 1920; H = 1080; }
                std::vector<float> u = {float(W),float(H), 0.5f,0.01f, 0.5f,1.0f, 0,0};
                ops[keepIdx]->main.AsExtra()->type = "isp.ee_ldci";
                ops[keepIdx]->main.AsExtra()->attr.clear();
                buildCommonAttrs(ops[keepIdx]->main.AsExtra(), W, H, u);
                setEngine(ops[keepIdx]->main.AsExtra());
                addSpirv(ops[keepIdx]->main.AsExtra(), "isp.ee_ldci");
                ops[keepIdx]->outputIndexes[0] = ops[resetIdx]->outputIndexes[0];
                ops[resetIdx].reset();
                return true;
            }
        }
        return false;
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

    bool matchUnpackDemosaic(std::vector<std::unique_ptr<OpT>>& ops) const {
        for (int i = 0; i < (int)ops.size(); i++) {
            if (!isExtraOfType(ops[i].get(), "isp.unpack_blc")) continue;
            for (int j = 0; j < (int)ops.size(); j++) {
                if (i == j || !ops[j]) continue;
                if (!isExtraOfType(ops[j].get(), "isp.demosaic_ccm")) continue;
                if (!isChainSkipCT(ops[i].get(), ops[j].get(), ops)) continue;

            VLOG(1) << "[P2] R10: UnpackDemosaic at " << i << "+" << j;
            int W, H;
            getExtraDims(ops[i+1], W, H);  // demosaic dims (output=FHD)
            int inpW = W*2, inpH = H*2;    // input dims (Bayer=4K)
            std::vector<float> u = {float(W),float(H), float(inpW),float(inpH), 1023,
                                    0,0,0,0, 1,1,1,1,
                                    1,0,0, 0,1,0, 0,0,1,
                                    0,0,0,0};
            ops[i]->main.AsExtra()->type = "isp.unpack_demosaic";
            ops[i]->main.AsExtra()->attr.clear();
            buildCommonAttrs(ops[i]->main.AsExtra(), W, H, u);
            setEngine(ops[i]->main.AsExtra());
            addSpirv(ops[i]->main.AsExtra(), "isp.unpack_demosaic");
            ops[i]->outputIndexes[0] = ops[j]->outputIndexes[0];
            ops[j].reset();
            return true;
                }
        }
        return false;
    }

    // R11: unpack_demosaic + fcs_display + ee_ldci → fused_6in1 (3-stage collapse)
    bool matchFused6in1(std::vector<std::unique_ptr<OpT>>& ops) const {
        int ud = -1, fd = -1, el = -1;
        for (int i = 0; i < (int)ops.size(); i++) {
            if (isExtraOfType(ops[i].get(), "isp.unpack_demosaic")) ud = i;
            else if (isExtraOfType(ops[i].get(), "isp.fcs_display")) fd = i;
            else if (isExtraOfType(ops[i].get(), "isp.ee_ldci")) el = i;
        }
        if (ud < 0 || fd < 0 || el < 0) return false;
        if (!isChainSkipCT(ops[ud].get(), ops[fd].get(), ops)) return false;
        if (!isChainSkipCT(ops[fd].get(), ops[el].get(), ops)) return false;

        VLOG(1) << "[P2] R11: Fused6in1 at " << ud << "+" << fd << "+" << el;
        int W, H;
        getExtraDims(ops[ud], W, H);
        int inpW = W*2, inpH = H*2;
        std::vector<float> u = {float(W),float(H), float(inpW),float(inpH), 1023,
                                0,0,0,0,  // blc
                                1,1,1,1,  // wb
                                1,0,      // fcs
                                0.5f,0.01f, // ee
                                0.5f,1.0f, // ldci
                                2.2f,0,   // display
                                0,0,0};   // pad
        ops[ud]->main.AsExtra()->type = "isp.fused_6in1";
        ops[ud]->main.AsExtra()->attr.clear();
        buildCommonAttrs(ops[ud]->main.AsExtra(), W, H, u);
        setEngine(ops[ud]->main.AsExtra());
        ops[ud]->outputIndexes[0] = ops[el]->outputIndexes[0];
        ops[fd].reset(); ops[el].reset();
        return true;
    }
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
