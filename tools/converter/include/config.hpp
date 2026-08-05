//
//  config.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <string>
#include <MNN/MNNDefine.h>
#include <fstream>
struct PostTreatContext;
class MNN_PUBLIC modelConfig {
public:
    modelConfig()
        : MNNModel(),
          prototxtFile(),
          modelFile(),
          bizCode("MNN"),
          model(modelConfig::MAX_SOURCE),
          saveHalfFloat(false){
    }
    ~ modelConfig ();
    enum MODEL_SOURCE { TENSORFLOW = 0, CAFFE, ONNX, MNN, TFLITE, TORCH, JSON, MAX_SOURCE };

    // MNN model path
    std::string MNNModel;
    // if model is tensorflow, this value is NULL;
    std::string prototxtFile;
    // tensorflow pb, or caffe model
    std::string modelFile;
    // bizCode
    std::string bizCode;
    // input config file
    std::string inputConfigFile;
    // model source
    MODEL_SOURCE model;
    bool saveHalfFloat;
    bool preserveInputType = false;
    bool forTraining = false;
    int weightQuantBits = 0;// If weightQuantBits > 0, it means the bit
    bool weightQuantAsymmetric = true;
    int weightQuantBlock = -1;
    bool useHQQ = false;
    // Bit-width for quant scale/zero-point storage. 32 = fp32, 16 = fp16; future: 8/4.
    int weightQuantScaleBit = 32;
    // The path of the model compression file that stores the int8 calibration table
    // or sparse parameters.
    std::string compressionParamsFile = "";
    bool saveStaticModel = false;
    int optimizePrefer = 0;
    float targetVersion = (float)MNN_VERSION_MAJOR + (float)MNN_VERSION_MINOR * 0.1f;
    int defaultBatchSize = 0;
    int optimizeLevel = 1;
    bool keepInputFormat = true;
    bool alignDenormalizedValue = true;
    bool detectSparseSpeedUp = false;
    bool convertMatmulToConv = true;
    bool useGeluApproximation = true;
    bool transformerFuse = false;
    bool transformerFuseC4 = true;
    bool allowCustomOp = false;
    bool groupConvNative = false;
    std::string customOpLibs = "";
    std::string authCode = "";
    std::string testDir = "";
    std::string testConfig;
    float testThredhold = 0.01;
    bool mnn2json = false;
    bool dumpInfo = false;
    bool saveExternalData = false;
    bool inSubGraph = false;
    // using external data when convert
    int64_t externalTreshold = 1024 * 64;
    std::ofstream* externalFile = nullptr;
    int64_t externalOffset = 0;
    bool useOriginRNNImpl = false;
    PostTreatContext* compressInfo = nullptr;
    bool splitQuantBlock = false;
    // Enable verbose output for each optimization pass (like LLVM's -debug-pass)
    bool dumpPass = false;
    // ISP fusion metadata propagated from the source model (ONNX metadata_props
    // key "isp_fusion"). Read by the IspChainFusion pass via Global<modelConfig>.
    // "enable" activates ISP Extra-op fusion; anything else keeps primitive ops.
    // This is a runtime switchable metadata field — no converter rebuild needed.
    std::string ispFusionMeta = "";
    // Pattern threshold: run try* with id <= threshold. 0=none, 41=basic, 999=all.
    int ispFusionThreshold = 999;
};

#endif // CONFIG_HPP
