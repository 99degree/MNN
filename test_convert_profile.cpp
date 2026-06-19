// test_convert_profile.cpp — convert ONNX->MNN + profile with CPU & Vulkan
// Usage:
//   LD_LIBRARY_PATH=./build_vk/OFF ./test_convert_profile model.onnx [size]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <stack>
#include <MNN/Interpreter.hpp>
#include <MNN/ErrorCode.hpp>

static bool run_cmd(const char* cmd) {
    int ret = system(cmd);
    if (ret == -1 || WEXITSTATUS(ret) != 0) {
        fprintf(stderr, "FAIL: %s (exit=%d)\n", cmd, WEXITSTATUS(ret));
        return false;
    }
    return true;
}

struct OpTiming {
    std::string name, type;
    float flops;
    double ms = 0;
    int count = 0;
    OpTiming() {}
    OpTiming(const std::string& n, const std::string& t, float f, double m, int c)
        : name(n), type(t), flops(f), ms(m), count(c) {}
};

static int profile_model(MNN::Interpreter* interpreter, int iterations, int inputSize,
                         MNNForwardType backend, const char* backendName) {
    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);

    MNN::ScheduleConfig config;
    config.type = backend;
    config.numThread = 4;

    // For Vulkan, hint to prefer GPU
    MNN::BackendConfig backendConfig;
    backendConfig.power = MNN::BackendConfig::Power_Normal;
    if (backend == MNN_FORWARD_VULKAN || backend == MNN_FORWARD_OPENCL) {
        backendConfig.precision = MNN::BackendConfig::Precision_Normal;
        config.backendConfig = &backendConfig;
    }

    auto* session = interpreter->createSession(config);
    if (!session) {
        fprintf(stderr, "  [%s] FAIL: createSession (backend not supported?)\n", backendName);
        return 1;
    }

    auto* input = interpreter->getSessionInput(session, nullptr);
    if (!input) {
        fprintf(stderr, "  [%s] FAIL: getSessionInput\n", backendName);
        return 1;
    }

    // Resize input
    std::vector<int> dims;
    int n = input->dimensions();
    if (n == 4) dims = {1, 1, inputSize, inputSize};
    else if (n == 3) dims = {1, inputSize, inputSize};
    else dims = std::vector<int>(n, inputSize);
    interpreter->resizeTensor(input, dims);
    interpreter->resizeSession(session);

    // Fill input
    if (input->host<void>() && input->size() > 0) {
        float* data = (float*)input->host<void>();
        for (int i = 0; i < input->size()/4; i++) data[i] = 0.5f;
    }

    // ── Aggregate profiling ──
    printf("\n  --- %s Aggregate (%d iters, %dx%d) ---\n", backendName, iterations, inputSize, inputSize);
    double totalMs = 0;
    for (int i = 0; i < iterations; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto err = interpreter->runSession(session);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err != MNN::NO_ERROR) {
            printf("  iter %d: FAIL error=%d\n", i, err);
            interpreter->releaseSession(session);
            return err;
        }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        totalMs += ms;
        printf("  iter %d: %.2f ms\n", i, ms);
    }
    printf("  avg: %.2f ms\n", totalMs / iterations);

    float mem=0, flops=0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &mem);
    interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    printf("  MEMORY: %.2f MB   FLOPS: %.2f M\n", mem, flops);

    // ── Per‑op profiling ──
    printf("\n  --- %s Per-Op ---\n", backendName);
    std::vector<OpTiming> opTimings;

    // Enumerate ops
    auto beforeEnum = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
        if (info) opTimings.emplace_back(info->name(), info->type(), info->flops(), 0, 0);
        return true;
    };
    auto afterDummy = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
        return true;
    };
    auto err = interpreter->runSessionWithCallBackInfo(session, beforeEnum, afterDummy);
    if (err != MNN::NO_ERROR) {
        printf("  FAIL: callbacks error=%d\n", err);
        interpreter->releaseSession(session);
        return err;
    }

    // Timed runs
    int nRuns = std::min(3, iterations);
    for (int r = 0; r < nRuns; r++) {
        std::stack<std::chrono::high_resolution_clock::time_point> opStack;
        int idx = 0;
        auto beforeT = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
            if (info) opStack.push(std::chrono::high_resolution_clock::now());
            return true;
        };
        auto afterT = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
            if (info && !opStack.empty()) {
                auto t1 = std::chrono::high_resolution_clock::now();
                auto t0 = opStack.top(); opStack.pop();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (idx < (int)opTimings.size()) {
                    opTimings[idx].ms += ms;
                    opTimings[idx].count++;
                }
                idx++;
            }
            return true;
        };
        err = interpreter->runSessionWithCallBackInfo(session, beforeT, afterT);
        if (err != MNN::NO_ERROR) {
            printf("  run %d: FAIL error=%d\n", r, err);
            break;
        }
    }

    // Print results
    printf("  %-40s %-18s %8s %10s %6s\n", "Op Name", "Type", "FLOPS(M)", "Time(ms)", "%");
    printf("  %s\n", std::string(80, '-').c_str());
    double sumMs = 0;
    for (auto& op : opTimings)
        sumMs += op.count > 0 ? op.ms / op.count : 0;
    for (auto& op : opTimings) {
        double avgOpMs = op.count > 0 ? op.ms / op.count : 0;
        double pct = sumMs > 0 ? 100.0 * avgOpMs / sumMs : 0;
        printf("  %-40s %-18s %8.2f %10.3f %5.1f%%\n",
               op.name.c_str(), op.type.c_str(), op.flops, avgOpMs, pct);
    }
    printf("  %s\n", std::string(80, '-').c_str());
    printf("  %-40s %-18s %8s %10.3f\n", "TOTAL", "", "", sumMs);

    interpreter->releaseSession(session);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.onnx [input_size]\n", argv[0]);
        return 1;
    }
    const char* onnxPath = argv[1];
    int inputSize = argc > 2 ? atoi(argv[2]) : 256;

    // ── Step 1: Convert ONNX -> MNN ──
    char mnnPath[1024];
    snprintf(mnnPath, sizeof(mnnPath), "%s/conv_%d.mnn",
        getenv("HOME") ? getenv("HOME") : ".", getpid());
    char cmd[2048];
    // Use MNNConvert from MNN_DIR, or try common locations
    const char* mnnDir = getenv("MNN_DIR");
    const char* converterPath = "./build/MNNConvert";
    char fallback[1024] = {0};
    if (mnnDir) {
        snprintf(fallback, sizeof(fallback), "%s/MNNConvert", mnnDir);
        converterPath = fallback;
    }
    snprintf(cmd, sizeof(cmd),
        "\"%s\" -f ONNX --modelFile \"%s\" --MNNModel \"%s\" --bizCode test 2>/dev/null",
        converterPath, onnxPath, mnnPath);

    printf("=== Converting ONNX -> MNN ===\n");
    if (!run_cmd(cmd)) return 1;
    printf("OK: %s -> %s\n", onnxPath, mnnPath);

    // ── Step 2: Load MNN model ──
    auto* interpreter = MNN::Interpreter::createFromFile(mnnPath);
    unlink(mnnPath);  // clean up temp file
    if (!interpreter) {
        fprintf(stderr, "FAIL: createFromFile\n");
        return 1;
    }
    printf("Model loaded OK\n");

    // ── Step 3: Profile ──
    // CPU backend first
    int ret = profile_model(interpreter, 3, inputSize, MNN_FORWARD_CPU, "CPU");

    // Vulkan backend second (if supported)
    if (ret == 0) {
        ret = profile_model(interpreter, 3, inputSize, MNN_FORWARD_VULKAN, "Vulkan");
    }

    MNN::Interpreter::destroy(interpreter);
    printf("\nDone.\n");
    return ret;
}
