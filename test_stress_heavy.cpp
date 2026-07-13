// test_stress_heavy.cpp - build a model with N Conv+Relu pairs via flatbuffers
// Then test debug profiling to reproduce the HEAVY model bug.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ErrorCode.hpp>
#include "MNN_generated.h"

static MNN::Interpreter* buildHeavyModel(int numPairs) {
    flatbuffers::FlatBufferBuilder fbb(1024);
    
    std::vector<flatbuffers::Offset<MNN::Op>> ops;
    std::vector<std::string> tensorNames;
    
    // Input tensor
    tensorNames.push_back("input");
    
    int currentOutput = 0;
    for (int p = 0; p < numPairs; p++) {
        int convIn = currentOutput;
        int convOut = currentOutput + 1;
        int reluOut = currentOutput + 2;
        
        // Conv weight (1x1x1x1)
        auto weightTensor = MNN::CreateBlob(
            fbb,
            MNN::DataType_DT_FLOAT,
            MNN::MNN_DATA_FORMAT_NCHW,
            {1, 1, 1, 1},
            fbb.CreateVector(std::vector<float>(1, 0.1f))
        );
        
        // Create OpT manually using flatbuffers
        auto conv2DCommon = MNN::CreateConvolution2DCommon(
            fbb, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, // padX/Y, kernelX/Y, strideX/Y
            1, 1, 0, 0, 0, 0, 1, MNN::ActivationFunction_RELU
        );
        
        auto convParam = MNN::CreateConvolution2D(
            fbb, conv2DCommon,
            fbb.CreateVector(std::vector<float>(1, 0.1f)),  // weight
            fbb.CreateVector(std::vector<float>(1, 0.0f)),  // bias
            0, weightTensor
        );
        
        auto convOp = MNN::CreateOp(
            fbb,
            MNN::OpType_Convolution,
            {},
            fbb.CreateVector(std::vector<int>({convIn})),
            fbb.CreateVector(std::vector<int>({convOut})),
            MNN::OpParameter_Convolution2D,
            convParam.Union(),
            0,
            fbb.CreateString(std::string("conv_") + std::to_string(p))
        );
        ops.push_back(convOp);
        
        auto reluOp = MNN::CreateOp(
            fbb,
            MNN::OpType_ReLU,
            {},
            fbb.CreateVector(std::vector<int>({convOut})),
            fbb.CreateVector(std::vector<int>({reluOut})),
            MNN::OpParameter_NONE,
            0,
            0,
            fbb.CreateString(std::string("relu_") + std::to_string(p))
        );
        ops.push_back(reluOp);
        
        tensorNames.push_back(std::string("conv_out_") + std::to_string(p));
        tensorNames.push_back(std::string("relu_out_") + std::to_string(p));
        currentOutput = reluOut;
    }
    
    // Convert tensor names to flatbuffer strings
    std::vector<flatbuffers::Offset<flatbuffers::String>> nameOffsets;
    for (auto& n : tensorNames) {
        nameOffsets.push_back(fbb.CreateString(n));
    }
    
    auto net = MNN::CreateNet(
        fbb,
        fbb.CreateVector(ops),
        fbb.CreateVector(nameOffsets),
        fbb.CreateVector(std::vector<int>({currentOutput})),  // outputIndexes
        0,  // buffer
        0,  // extraTensorDescribe
        0,  // tensorNumber
        0,  // inputName
        0,  // outputName
        0,  // usage
        -1, // version
        0,  // opaque
        0,  // inputTensorIds
        0   // outputTensorIds
    );
    
    fbb.Finish(net);
    
    return MNN::Interpreter::createFromBuffer(fbb.GetBufferPointer(), fbb.GetSize());
}

int main(int argc, char* argv[]) {
    int numPairs = 100;
    if (argc > 1) numPairs = atoi(argv[1]);
    
    fprintf(stderr, "Building model with %d Conv+Relu pairs...\n", numPairs);
    auto* interpreter = buildHeavyModel(numPairs);
    if (!interpreter) {
        fprintf(stderr, "FAIL: buildHeavyModel returned null\n");
        return 1;
    }
    fprintf(stderr, "Interpreter created OK\n");
    
    // Session_Debug mode
    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);
    
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 2;
    
    auto* session = interpreter->createSession(config);
    if (!session) {
        fprintf(stderr, "FAIL: createSession returned null\n");
        return 1;
    }
    fprintf(stderr, "Session created OK\n");
    
    auto* input = interpreter->getSessionInput(session, nullptr);
    if (input) {
        interpreter->resizeTensor(input, {1, 1, 64, 64});
        float* data = (float*)input->host<void>();
        if (data) {
            for (int i = 0; i < 64*64; i++) data[i] = 0.5f;
        }
    }
    
    // Test 1: Aggregate
    fprintf(stderr, "\n--- Test 1: runSession (aggregate) ---\n");
    auto err = interpreter->runSession(session);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSession error=%d\n", err);
        interpreter->releaseSession(session);
        MNN::Interpreter::destroy(interpreter);
        return 1;
    }
    float mem=0, flops=0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &mem);
    interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    fprintf(stderr, "OK: mem=%.3f flops=%.3f\n", mem, flops);
    
    // Test 2: Per-node (the one that should trigger COMPUTE_SIZE_ERROR for large models)
    fprintf(stderr, "\n--- Test 2: runSessionWithCallBackInfo ---\n");
    int beforeCount=0, afterCount=0;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& t, const MNN::OperatorInfo* info) -> bool {
        beforeCount++;
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& t, const MNN::OperatorInfo* info) -> bool {
        afterCount++;
        return true;
    };
    
    err = interpreter->runSessionWithCallBackInfo(session, before, after);
    if (err != MNN::NO_ERROR) {
        fprintf(stderr, "FAIL: runSessionWithCallBackInfo error=%d\n", err);
    } else {
        int expected = (numPairs * 2) + 1; // input + conv + relu per pair + raster
        fprintf(stderr, "OK: before=%d after=%d (expected ~%d)\n", beforeCount, afterCount, expected);
    }
    
    // Test 3: Second aggregate to confirm session not stuck
    fprintf(stderr, "\n--- Test 3: runSession again ---\n");
    err = interpreter->runSession(session);
    fprintf(stderr, "result=%d\n", err);
    
    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    fprintf(stderr, "\nDone.\n");
    return 0;
}
