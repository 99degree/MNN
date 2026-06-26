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

static bool isChain(const OpT* a, const OpT* b) {
    return a && b && !a->outputIndexes.empty() && !b->inputIndexes.empty()
           && a->outputIndexes[0] == b->inputIndexes[0];
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

    // R4: Conv(3×3,unsharp) → isp.ee
    bool tryEe(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Convolution) return false;
        auto* c = ops[i]->main.AsConvolution2D();
        if (!isEeConv(c)) return false;

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

    // R5: Pool(AVG,3×3) + Sub + Mul + Add → isp.ldci
    bool tryLdci(std::vector<std::unique_ptr<OpT>>& ops, int& i) const {
        if (ops[i]->type != MNN::OpType_Pooling) return false;
        auto* p = ops[i]->main.AsPool();
        if (!isAvgPool3x3(p)) return false;

        if (i+3 >= (int)ops.size()) return false;
        if (!isBinaryType(ops[i+1].get(), MNN::BinaryOpOperation_SUB)) return false;
        if (!isBinaryType(ops[i+2].get(), MNN::BinaryOpOperation_MUL)) return false;
        if (!isBinaryType(ops[i+3].get(), MNN::BinaryOpOperation_ADD)) return false;
        if (!isChain(ops[i].get(), ops[i+1].get()) ||
            !isChain(ops[i+1].get(), ops[i+2].get()) ||
            !isChain(ops[i+2].get(), ops[i+3].get())) return false;

        std::vector<float> u = {float(mW),float(mH),0.5f,1.0f, 0,0,0,0};

        ops[i]->type = MNN::OpType_Extra; ops[i]->main.type = MNN::OpParameter_Extra;
        auto* ex = new MNN::ExtraT(); ex->type = "isp.ldci";
        buildCommonAttrs(ex, mW, mH, u);
        setEngine(ex);
        addSpirv(ex, "isp.ldci");
        ops[i]->main.value = ex;
        ops[i]->outputIndexes[0] = ops[i+3]->outputIndexes[0];

        ops[i+1].reset(); ops[i+2].reset(); ops[i+3].reset();
        i += 3;
        VLOG(2) << "[P1] R5: ldci at " << i-3 << " (4 ops fused)";
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

        std::vector<float> u = {float(mW),float(mH),0,1,2.2f,1, 0,0};

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
        for (int i = 0; i+1 < (int)ops.size(); i++) {
            if (!isExtraOfType(ops[i].get(), "isp.fcs") ||
                !isExtraOfType(ops[i+1].get(), "isp.display") ||
                !isChain(ops[i].get(), ops[i+1].get())) continue;

            VLOG(1) << "[P2] R8: FcsDisplay at " << i;
            auto* fcs = ops[i]->main.AsExtra();
            // Extract fcs strength from const buffer
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
            ops[i]->outputIndexes[0] = ops[i+1]->outputIndexes[0];
            ops[i+1].reset(); i++;
            return true;
        }
        return false;
    }

    // R9: isp.ee + isp.ldci → isp.ee_ldci
    bool matchEeLdci(std::vector<std::unique_ptr<OpT>>& ops) const {
        for (int i = 0; i+1 < (int)ops.size(); i++) {
            if (!isExtraOfType(ops[i].get(), "isp.ee") ||
                !isExtraOfType(ops[i+1].get(), "isp.ldci") ||
                !isChain(ops[i].get(), ops[i+1].get())) continue;

            VLOG(1) << "[P2] R9: EeLdci at " << i;
            int W, H;
            getExtraDims(ops[i], W, H);
            std::vector<float> u = {float(W),float(H), 0.5f,0.01f, 0.5f,1.0f, 0,0};
            ops[i]->main.AsExtra()->type = "isp.ee_ldci";
            ops[i]->main.AsExtra()->attr.clear();
            buildCommonAttrs(ops[i]->main.AsExtra(), W, H, u);
            setEngine(ops[i]->main.AsExtra());
            addSpirv(ops[i]->main.AsExtra(), "isp.ee_ldci");
            ops[i]->outputIndexes[0] = ops[i+1]->outputIndexes[0];
            ops[i+1].reset(); i++;
            return true;
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
        for (int i = 0; i+1 < (int)ops.size(); i++) {
            if (!isExtraOfType(ops[i].get(), "isp.unpack_blc") ||
                !isExtraOfType(ops[i+1].get(), "isp.demosaic_ccm") ||
                !isChain(ops[i].get(), ops[i+1].get())) continue;

            VLOG(1) << "[P2] R10: UnpackDemosaic at " << i;
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
            ops[i]->outputIndexes[0] = ops[i+1]->outputIndexes[0];
            ops[i+1].reset(); i++;
            return true;
        }
        return false;
    }

    // R11: unpack_demosaic + fcs_display + ee_ldci → fused_6in1 (3-stage collapse)
    bool matchFused6in1(std::vector<std::unique_ptr<OpT>>& ops) const {
        for (int i = 0; i+2 < (int)ops.size(); i++) {
            if (!isExtraOfType(ops[i].get(),   "isp.unpack_demosaic") ||
                !isExtraOfType(ops[i+1].get(), "isp.fcs_display") ||
                !isExtraOfType(ops[i+2].get(), "isp.ee_ldci") ||
                !isChain(ops[i].get(), ops[i+1].get()) ||
                !isChain(ops[i+1].get(), ops[i+2].get())) continue;

            VLOG(1) << "[P2] R11: Fused6in1 at " << i;
            int W, H;
            getExtraDims(ops[i], W, H);
            int inpW = W*2, inpH = H*2;
            std::vector<float> u = {float(W),float(H), float(inpW),float(inpH), 1023,
                                    0,0,0,0,  // blc
                                    1,1,1,1,  // wb
                                    1,0,      // fcs
                                    0.5f,0.01f, // ee
                                    0.5f,1.0f, // ldci
                                    2.2f,0,   // display
                                    0,0,0};   // pad
            ops[i]->main.AsExtra()->type = "isp.fused_6in1";
            ops[i]->main.AsExtra()->attr.clear();
            buildCommonAttrs(ops[i]->main.AsExtra(), W, H, u);
            setEngine(ops[i]->main.AsExtra());
            // addSpirv for fused_6in1 if available
            ops[i]->outputIndexes[0] = ops[i+2]->outputIndexes[0];
            ops[i+1].reset(); ops[i+2].reset(); i += 2;
            return true;
        }
        return false;
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

// ── Registration ──
// Registers as "IspChainFusion" so optimizer finds it via optimizeNet()
static PostConverterRegister<IspChainFusion> __isp_fusion("IspChainFusion");
