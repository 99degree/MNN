// IspFormatCleanup.cpp — post-format-converter cleanup for direct-emit isp.* ops.
//
// The ONNX importer emits primitive ops and pass0 direct-emit isp.* Extra ops
// (no MNN→MNN chain fusion — that path was removed). Two cleanup passes live
// here and run in the afterProgramConvert phase of optimizeNet():
//
//   RemoveExtraConvertTensor — removes ConvertTensor ops inserted by
//     AddTensorFormatConverter between Extra (isp.*) ops; the inserted
//     NC4HW4↔NCHW layout change breaks the VulkanFuse Extra chain because
//     the custom SPIR-V bindings copy raw planes.
//
//   NchwTagNormalize — normalizes defaultDimentionFormat to NCHW for every
//     non-Input op (the Op schema defaults to NHWC and the importer never
//     sets it for primitive ops, which makes primitive consumers of isp.*
//     outputs shape-infer nonsense channel counts).

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN_generated.h>

#include "../Global.hpp"
#include "../PostTreatUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"

namespace MNN {

// ── Post-format-converter cleanup ──
// Removes ConvertTensor ops inserted by AddTensorFormatConverter between
// Extra (isp.*) ops — these break the VulkanFuse Extra chain because the
// conversion inserts an NC4HW4↔NCHW layout change the custom SPIR-V doesn't
// handle. Runs AFTER AddTensorFormatConverter in the converter pipeline.
class RemoveExtraConvertTensor : public PostConverter {
public:
    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
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

// ── Dimension-format normalization ──
// The Op schema defaults defaultDimentionFormat to NHWC, and the ONNX
// importer never sets it for primitive ops, so every imported op INHERITS
// the schema's NHWC default — a debayer output declared [1,3,H,W] is then
// shape-interpreted as NHWC (C=128!) and any primitive consumer
// (Conv/Cast/Gather) gets a nonsense NC4HW4 conversion
// ("inputChannel: 128 ... Compute Shape Error").
//
// isp.* Extra ops bypass format tags inside VulkanFuse (bindings copy raw
// planes), and their outputs are physically NCHW planes per the Kotlin
// valueInfo declarations. So: tag every non-Input op NCHW (Input ops keep
// the importer-assigned NCHW tag).
class NchwTagNormalize : public PostConverter {
public:
    bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& ops = net->oplists;
        bool changed = false;
        for (auto& op : ops) {
            if (!op) continue;
            if (op->type == MNN::OpType_Input) continue;
            if (op->defaultDimentionFormat != MNN_DATA_FORMAT_NCHW) {
                op->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
                changed = true;
            }
        }
        return changed;
    }
};
static PostConverterRegister<NchwTagNormalize> __nchw_tag_norm("NchwTagNormalize");

} // namespace MNN
