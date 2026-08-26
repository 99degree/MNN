// MNNConvertBuffer.cpp — C-only memory-buffer → memory-buffer conversion API.
//
// Converts an ONNX model held entirely in memory to an MNN flatbuffer held
// entirely in memory: no temp files, no memfd, no /proc/self/fd, no disk
// round-trip. The pipeline mirrors MNN::Cli::convertModel(ONNX) +
// writeFb() but operates on buffers:
//
//   onnx bytes ──ParseFromArray──▶ onnx::ModelProto
//        ──onnx2MNNNet──▶ MNN::NetT
//        ──optimizeNet (optimizeLevel=2, incl. IspChainFusion)──▶ NetT'
//        ──postTreat + MNN::Net::Pack──▶ MNN flatbuffer bytes
//
// The caller frees the returned buffer with free().
//
// Thread-safety: MNN's converter is NOT thread-safe (Global<modelConfig>
// stores a pointer to a per-call stack-local config, so concurrent
// conversions race on the pointee → SIGSEGV inside optimizeNet). Every
// conversion entry point in this file and in mnn_convert_api.cpp serializes
// under the single g_convertMutex (exported as mnn_convert_lock/unlock).

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "MNN_generated.h"
#include "config.hpp"
#include "../optimizer/Global.hpp"
#include "PostConverter.hpp"
#include "CommonUtils.hpp"
#include "onnx.pb.h"
#include "onnxConverter.hpp"

// Defined in writeFb.cpp (not declared in a public header).
int postTreat(std::unique_ptr<MNN::NetT>& netT, const modelConfig& config);

namespace MNN {
namespace BufferConvert {

static std::mutex g_convertMutex;

// Make Input op order the same as the origin model (mirror of the
// file-static _reorderInputs in cli.cpp).
static void reorderInputs(const std::vector<std::string>& inputNames, MNN::NetT* netT) {
    if (inputNames.empty()) {
        return;
    }
    auto oplists = std::move(netT->oplists);
    std::vector<std::unique_ptr<MNN::OpT>> inputOps;
    for (auto& op : oplists) {
        if (nullptr == op.get()) {
            continue;
        }
        if (op->type != MNN::OpType_Input || op->outputIndexes.empty()) {
            continue;
        }
        inputOps.emplace_back(std::move(op));
    }
    for (int i = 0; i < (int)inputNames.size(); ++i) {
        for (auto& op : inputOps) {
            if (nullptr == op.get()) {
                continue;
            }
            if (netT->tensorName[op->outputIndexes[0]] == inputNames[i]) {
                netT->oplists.emplace_back(std::move(op));
                break;
            }
        }
    }
    for (auto& op : oplists) {
        if (nullptr != op.get()) {
            netT->oplists.emplace_back(std::move(op));
        }
    }
}

/// Serialize a NetT to MNN flatbuffer bytes (malloc'd, caller frees).
/// Mirrors writeFb() minus the file write and minus LOG(FATAL) on
/// unsupported ops (returns -1 instead).
static int serializeNet(std::unique_ptr<MNN::NetT>& netT, const modelConfig& config,
                        std::unique_ptr<MNN::OpT>&& metaOp, void** out_data, size_t* out_size) {
    if (postTreat(netT, config) != 0) {
        return -1;
    }
    // Merge Meta op into metaOp
    auto oplist = std::move(netT->oplists);
    for (auto& op : oplist) {
        if (op->type == OpType_Extra) {
            auto dstExtra = metaOp->main.AsExtra();
            auto extra = op->main.AsExtra();
            if (extra->type == "Meta" && extra->engine == "MNN") {
                for (auto& attr : extra->attr) {
                    dstExtra->attr.emplace_back(std::move(attr));
                }
                continue;
            }
        }
        netT->oplists.emplace_back(std::move(op));
    }
    // Detect unsupported ops (non-MNN Extra engines)
    for (const auto& op : netT->oplists) {
        if (op->type == MNN::OpType_Extra) {
            if (op->main.AsExtra()->engine != "MNN") {
                return -1;
            }
        }
    }
    // Add version info
    netT->extraInfo.reset(new ExtraInfoT);
    netT->extraInfo->version = MNN_VERSION;
    if (metaOp->main.AsExtra()->attr.size() > 0) {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Extra::Pack(builder, metaOp->main.AsExtra()));
        netT->extraInfo->buffer.resize(builder.GetSize());
        ::memcpy(netT->extraInfo->buffer.data(), builder.GetBufferPointer(), builder.GetSize());
    }
    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);
    size_t sizeOutput = builderOutput.GetSize();
    auto* bufferOutput = builderOutput.GetBufferPointer();

    void* copy = malloc(sizeOutput > 0 ? sizeOutput : 1);
    if (copy == nullptr) {
        return -1;
    }
    ::memcpy(copy, bufferOutput, sizeOutput);
    *out_data = copy;
    *out_size = sizeOutput;
    return 0;
}

} // namespace BufferConvert
} // namespace MNN

extern "C" {

MNN_PUBLIC void mnn_convert_lock() { MNN::BufferConvert::g_convertMutex.lock(); }
MNN_PUBLIC void mnn_convert_unlock() { MNN::BufferConvert::g_convertMutex.unlock(); }

/// Convert an ONNX model from a memory buffer to an MNN model in a memory
/// buffer. Pure C ABI (extern "C", malloc/free, no C++ types across the
/// boundary) — callable directly from Rust FFI without the converter CLI.
///
/// @param onnx_data    ONNX protobuf bytes
/// @param onnx_len     byte length of onnx_data
/// @param out_data     [out] malloc'd MNN flatbuffer bytes (caller frees with free())
/// @param out_size     [out] byte length of *out_data
/// @param error_msg    [out] optional error string buffer (may be NULL)
/// @param error_msg_cap size of error_msg
/// @return 0 on success, -1 on failure (error_msg set when provided)
MNN_PUBLIC int mnn_convert_onnx_to_mnn_buffer(const void* onnx_data, size_t onnx_len,
                                              void** out_data, size_t* out_size,
                                              char* error_msg, size_t error_msg_cap) {
    std::lock_guard<std::mutex> convertLock(MNN::BufferConvert::g_convertMutex);
    auto fail = [&](const char* msg) {
        if (error_msg && error_msg_cap > 0) {
            snprintf(error_msg, error_msg_cap, "%s", msg);
        }
        if (out_data) *out_data = nullptr;
        if (out_size) *out_size = 0;
        return -1;
    };
    if (onnx_data == nullptr || onnx_len == 0 || out_data == nullptr || out_size == nullptr) {
        return fail("NULL/empty ONNX data or NULL out pointers");
    }

    // Parse ONNX protobuf from memory.
    onnx::ModelProto onnxModel;
    if (!onnxModel.ParseFromArray(onnx_data, (int)onnx_len) || onnxModel.graph().node_size() == 0) {
        return fail("failed to parse ONNX protobuf from memory");
    }

    modelConfig config;
    config.model = modelConfig::ONNX;
    config.bizCode = "MNN";
    config.optimizeLevel = 2;  // full pipeline incl. IspChainFusion
    config.weightQuantBits = 0;
    config.saveHalfFloat = false;
    // ISP wire format is packed-Bayer quad-NHWC [1,H/2,W/2,4]. Declare the
    // 4-D graph inputs NHWC so runtime session tensors match and host→
    // device uploads stay verbatim (no NCHW↔NHWC reorder scramble).
    config.onnxInputNHWC = true;
    config.preserveInputType = true;
    config.saveExternalData = false;
    // PostTreatContext for postTreat() (same as Cli::convertModel).
    CommonKit::loadCompress(config);

    // Register config so onnxOpConverter::convertDataType honors
    // preserveInputType and the postconvert passes see the source type.
    Global<modelConfig>::Reset(&config);

    std::unique_ptr<MNN::NetT> netT(new MNN::NetT());
    std::unique_ptr<MNN::OpT> metaOp(new MNN::OpT);
    metaOp->type = MNN::OpType_Extra;
    metaOp->main.value = new MNN::ExtraT;
    metaOp->main.type = MNN::OpParameter_Extra;
    metaOp->main.AsExtra()->type = "Meta";
    metaOp->main.AsExtra()->engine = "MNN";

    std::vector<std::string> inputNames;
    std::string modelDir;  // no external-data files in buffer mode
    if (onnx2MNNNet(onnxModel, modelDir, config.bizCode, netT, metaOp.get(), inputNames) != 0) {
        return fail("onnx2MNNNet failed");
    }
    if (netT.get() == nullptr || netT->oplists.empty()) {
        return fail("empty MNN net after parse");
    }

    std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, config.forTraining, config, {});
    if (newNet.get() == nullptr) {
        return fail("optimizeNet returned NULL");
    }
    MNN::BufferConvert::reorderInputs(inputNames, newNet.get());
    if (MNN::BufferConvert::serializeNet(newNet, config, std::move(metaOp), out_data, out_size) != 0) {
        return fail("serialize MNN net failed");
    }
    return 0;
}

} // extern "C"
