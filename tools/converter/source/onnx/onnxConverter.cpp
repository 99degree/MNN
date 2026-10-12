//
//  onnxConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include <queue>

#include "MNN_generated.h"
#include "OnnxUtils.hpp"
#include "logkit.h"
#include "core/IspOpsetValidate.hpp"

#include "OnnxTmpGraph.hpp"
#include "../optimizer/Global.hpp"
#include "config.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "onnx.pb.h"
#include "onnxConverter.hpp"
#include "onnxOpConverter.hpp"

int onnx2MNNNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<MNN::NetT>& netT, MNN::OpT* meta, std::vector<std::string>& inputNames) {
    std::string modelDir;
    size_t pos = inputModel.find_last_of("\\/");
    if (pos != std::string::npos) {
        modelDir = inputModel.substr(0, pos + 1);
    }

    onnx::ModelProto onnxModel;
    // read ONNX Model
    bool success = onnx_read_proto_from_binary(inputModel.c_str(), &onnxModel);
    DCHECK(success) << "read onnx model failed: " << inputModel;
    if (!success) {
        MNN_ERROR("[ERROR] Model file is not onnx model.\n");
        return 1;
    }
    return onnx2MNNNet(onnxModel, modelDir, bizCode, netT, meta, inputNames);
}

/// Buffer-based entry: convert an already-parsed ONNX ModelProto to a MNN
/// NetT entirely in memory (no model file, no external-data directory).
int onnx2MNNNet(onnx::ModelProto& onnxModel, const std::string& modelDir, const std::string bizCode,
                std::unique_ptr<MNN::NetT>& netT, MNN::OpT* meta, std::vector<std::string>& inputNames) {
    int opsetVersion = 13;
    auto opsetInfo = onnxModel.opset_import();
    if (!opsetInfo.empty()) {
        opsetVersion = static_cast<int>(opsetInfo.begin()->version());
    }
    LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();
    LOG(INFO) << "ONNX Model opset version: " << opsetVersion;

    const auto& onnxGraph = onnxModel.graph();
    const int nodeCount   = onnxGraph.node_size();
    if (0 == nodeCount) {
        MNN_ERROR("[ERROR] Invalid ONNX Model (no nodes).\n");
        return 1;
    }
    for (int i=0; i<onnxGraph.input_size(); ++i) {
        inputNames.emplace_back(onnxGraph.input(i).name());
    }
    std::unique_ptr<OnnxScope> scope(new OnnxScope(&onnxGraph, netT.get(), modelDir));
    scope->mOpsetVersion = opsetVersion;
    // find the inputs which do not have initializer
    const auto& initializers         = scope->mInitializers;
    const auto& inputs               = scope->mInputs;
    const auto& outputs              = scope->mOutputs;
    // set input node to MNN net
    for (const auto& iter : inputs) {
        bool notHaveInitializer = initializers.find(iter.first) == initializers.end();
        if (notHaveInitializer) {
            MNN::OpT* MNNOp  = new MNN::OpT;
            MNNOp->name      = iter.first;
            MNNOp->type      = MNN::OpType_Input;
            MNNOp->main.type = MNN::OpParameter_Input;
            auto inputParam  = new MNN::InputT;
            const auto it    = inputs.find(iter.first);
            //FUNC_PRINT_ALL(iter.first.c_str(), s);
            DCHECK(it != inputs.end()) << "Input Paramter ERROR ==> " << iter.first;
            const auto& tensorInfo = (it->second)->type().tensor_type();
            const int inputDimSize = tensorInfo.shape().dim_size();
            inputParam->dims.resize(inputDimSize);
            for (int i = 0; i < inputDimSize; ++i) {
                const auto& dim = tensorInfo.shape().dim(i);
                if (dim.has_dim_value()) {
                    inputParam->dims[i] = static_cast<int32_t>(dim.dim_value());
                } else {
                    inputParam->dims[i] = -1;
                }
            }
            inputParam->dtype   = onnxOpConverter::convertDataType(tensorInfo.elem_type());
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
            MNNOp->outputIndexes.push_back(scope->declareTensor(iter.first));
            MNNOp->main.value = inputParam;
            netT->oplists.emplace_back(MNNOp);
        }
    }

    // onnx model not all topo sort graph, sort it
    std::vector<int> idxMap = OnnxScope::topoSort(onnxGraph);

    auto makeConst = [&](const std::string& inputName) {
        const auto it         = initializers.find(inputName);
        if (it != initializers.end() && scope->lookupTensor(it->first) == -1) {
            // Create const Op
            MNN::OpT* constOp   = new MNN::OpT;
            constOp->type       = MNN::OpType_Const;
            constOp->main.type  = MNN::OpParameter_Blob;
            constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second, modelDir, constOp);
            constOp->name    = it->first;
            constOp->outputIndexes.push_back(scope->declareTensor(it->first));
            netT->oplists.emplace_back(constOp);
            scope->insertConstant(inputName, constOp);
        }
    };
    for (int i=0; i<onnxGraph.output_size(); ++i) {
        makeConst(onnxGraph.output(i).name());
    }
    // Declare all outputs
    for (int idx = 0; idx < nodeCount; ++idx) {
        int i = idxMap.size() == nodeCount ? idxMap[idx] : idx;
        const auto& onnxNode = onnxGraph.node(i);
        for (int k = 0; k < onnxNode.output_size(); k++) {
            scope->declareTensor(onnxNode.output(k));
        }
    }

    // onnx node ==> MNN node
    for (int idx = 0; idx < nodeCount; ++idx) {
        int i = idxMap.size() == nodeCount ? idxMap[idx] : idx;
        const auto& onnxNode = onnxGraph.node(i);
        const auto& opType   = onnxNode.op_type();
        // [ISP OPSET TRAP] Remember that this graph is ISP-bearing even if the
        // isp.* nodes are later dropped/rewritten (domain may be carried on the
        // node instead of the op_type prefix).
        if (opType.rfind("isp.", 0) == 0 || onnxNode.domain() == "isp") {
            MNN::ispModelFlag() = true;
        }

        // name maybe null, use the first output name as node-name
        const auto& name = onnxNode.output(0);
        auto opConverter = onnxOpConverterSuit::get()->search(opType);

        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = name;
        MNNOp->type      = opConverter->opType();
        MNNOp->main.type = opConverter->type();

        // convert initializer to be Constant node(op)
        for (int k = 0; k < onnxNode.input_size(); ++k) {
            const auto& inputName = onnxNode.input(k);
            makeConst(inputName);
        }

        // build input and output
        for (int k = 0; k < onnxNode.input_size(); k++) {
            int inputIdx = scope->lookupTensor(onnxNode.input(k));
            if (inputIdx < 0) {
                LOG(INFO) << "Check it out ==> " << MNNOp->name << " has empty input, the index is " << k;
            }
            MNNOp->inputIndexes.push_back(inputIdx);
        }
        for (int k = onnxNode.input_size() - 1; k >= 0 && MNNOp->inputIndexes[k] < 0; --k) {
            MNNOp->inputIndexes.pop_back();
        }
        for (int k = 0; k < onnxNode.output_size(); k++) {
            MNNOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
        }
        // build op
        opConverter->run(MNNOp, &onnxNode, scope.get());
        // [MNN-ISP-CONVERT] Trace isp.* domain ops as they are lowered from ONNX
        // to MNN Extra, and validate the emitted MNN isp.* name against the
        // canonical IspSpvLookup table. Catches name drift (e.g. "isp.algogamma"
        // vs "isp.algo_gamma") at conversion time, before VulkanFuse ever sees it.
        if (opType.rfind("isp.", 0) == 0) {
            MNN::ispModelFlag() = true;
            const char* _mnnEngine = "<n/a>";
            std::string _mnnTypeStr = "<n/a>";
            if (MNNOp->main.type == MNN::OpParameter_Extra && MNNOp->main.value != nullptr) {
                auto* _ex = reinterpret_cast<MNN::ExtraT*>(MNNOp->main.value);
                _mnnEngine  = _ex->engine.c_str();
                _mnnTypeStr = _ex->type;
            }
            fprintf(stderr,
                    "[MNN-ISP-CONVERT] onnx domain=%s op_type=%s -> mnn OpType=%d OpParam=%d engine=%s type=%s name=%s\n",
                    onnxNode.domain().c_str(), opType.c_str(), (int)MNNOp->type,
                    (int)MNNOp->main.type, _mnnEngine, _mnnTypeStr.c_str(), MNNOp->name.c_str());
            fflush(stderr);
            if (!MNN::isKnownIspOp(_mnnTypeStr)) {
                fprintf(stderr,
                        "[MNN-ISP-TRAP] INVALID isp.* opset at convert: onnx op_type=%s -> mnn type=%s "
                        "(no matching IspSpvLookup shader)\n",
                        opType.c_str(), _mnnTypeStr.c_str());
                fflush(stderr);
                if (MNN::ispOpsetTrapEnabled()) {
                    fprintf(stderr, "[MNN-ISP-TRAP] MNN_ISP_TRAP=1 -> aborting conversion\n");
                    fflush(stderr);
                    abort();
                }
            }
        }
        if (MNNOp->type == MNN::OpType_Const) {
            scope->insertConstant(name, MNNOp);
        }
        netT->oplists.emplace_back(MNNOp);
    }
    netT->tensorNumber = netT->tensorName.size();
    // set MNN net output name
    for (int i = 0; i < onnxGraph.output_size(); ++i) {
        const auto& output = onnxGraph.output(i);
        netT->outputName.emplace_back(output.name());
    }

    netT->sourceType = MNN::NetSource_ONNX;
    netT->bizCode    = bizCode;
    // [ISP OPSET TRAP] MISSING: this ONNX graph declared isp-domain nodes, but
    // none survived as isp.* Extra ops in the emitted MNN model. Every ISP block
    // degraded to CPU primitives — report it here (and let the session-time trap
    // enforce it) instead of silently shipping a CPU-only model.
    {
        int ispOpEmitted = 0;
        for (const auto& op : netT->oplists) {
            if (op != nullptr && op->type == MNN::OpType_Extra && op->main.type == MNN::OpParameter_Extra &&
                op->main.value != nullptr) {
                auto* ex = reinterpret_cast<MNN::ExtraT*>(op->main.value);
                if (ex->type.rfind("isp.", 0) == 0) {
                    ++ispOpEmitted;
                }
            }
        }
        if (MNN::ispModelFlag() && ispOpEmitted == 0) {
            fprintf(stderr,
                    "[MNN-ISP-TRAP] MISSING isp.* opset at convert: ONNX graph had isp-domain nodes but "
                    "the MNN model contains 0 isp.* Extra ops\n");
            fflush(stderr);
            if (MNN::ispOpsetTrapEnabled()) {
                fprintf(stderr, "[MNN-ISP-TRAP] MNN_ISP_TRAP=1 -> aborting conversion\n");
                fflush(stderr);
                abort();
            }
        } else {
            fprintf(stderr, "[MNN-ISP-CONVERT] isp-domain model=%d isp Extra ops emitted=%d\n",
                    MNN::ispModelFlag() ? 1 : 0, ispOpEmitted);
            fflush(stderr);
        }
    }
    auto metaSize = onnxModel.metadata_props_size();
    for (int i=0; i<metaSize; ++i) {
        std::unique_ptr<MNN::AttributeT> dstMeta(new MNN::AttributeT);
        auto srcMeta = onnxModel.metadata_props(i);
        dstMeta->key = srcMeta.key();
        dstMeta->s = srcMeta.value();
        meta->main.AsExtra()->attr.emplace_back(std::move(dstMeta));
    }

    return 0;
}
