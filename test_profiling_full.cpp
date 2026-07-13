// test_profiling_full.cpp — aggregate + per‑op profiling with actual timing
// Usage: LD_LIBRARY_PATH=./build_vk/OFF ./test_profiling_full model.mnn [iterations]
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <string>
#include <stack>
#include <MNN/Interpreter.hpp>
#include <MNN/ErrorCode.hpp>

struct OpTiming {
    std::string name;
    std::string type;
    float flops;
    double ms = 0;   // accumulated time in ms
    int count = 0;   // number of times this op was measured
    OpTiming() {}
    OpTiming(const std::string& n, const std::string& t, float f, double m, int c)
        : name(n), type(t), flops(f), ms(m), count(c) {}
};

int main(int argc, char* argv[]) {
    const char* modelPath = argc > 1 ? argv[1] : nullptr;
    int iterations = argc > 2 ? atoi(argv[2]) : 3;
    if (!modelPath) {
        fprintf(stderr, "Usage: %s model.mnn [iterations]\n", argv[0]);
        return 1;
    }

    // ── Load model ──
    auto* interpreter = MNN::Interpreter::createFromFile(modelPath);
    if (!interpreter) { fprintf(stderr, "FAIL: createFromFile\n"); return 1; }

    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;

    auto* session = interpreter->createSession(config);
    if (!session) { fprintf(stderr, "FAIL: createSession\n"); return 1; }

    auto* input = interpreter->getSessionInput(session, nullptr);
    if (input && input->host<void>() && input->size() > 0) {
        float* data = (float*)input->host<void>();
        for (int i = 0; i < input->size()/4; i++) data[i] = 0.5f;
    }

    // ── 1. Aggregate profiling ──
    printf("\n=== AGGREGATE PROFILING (%d iterations) ===\n", iterations);
    double totalMs = 0;
    for (int i = 0; i < iterations; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto err = interpreter->runSession(session);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err != MNN::NO_ERROR) {
            printf("  iter %d: FAIL error=%d\n", i, err);
            return 1;
        }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        totalMs += ms;
        printf("  iter %d: %.2f ms\n", i, ms);
    }
    double avgMs = totalMs / iterations;
    printf("  avg: %.2f ms\n", avgMs);

    // ── 2. Session info ──
    float mem=0, flops=0;
    interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &mem);
    interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    printf("\n=== SESSION INFO ===\n");
    printf("  MEMORY: %.2f MB\n", mem);
    printf("  FLOPS:  %.2f M\n", flops);

    // ── 3. Per‑op profiling with actual timer ──
    printf("\n=== PER‑OP PROFILING (actual timing via callbacks) ===\n");

    // First, enumerate all ops with a warm-up run
    std::vector<OpTiming> opTimings;
    {
        auto beforeEnum = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
            if (info) opTimings.emplace_back(info->name(), info->type(), info->flops(), 0, 0);
            return true;
        };
        auto afterEnum = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
            return true;
        };
        auto err = interpreter->runSessionWithCallBackInfo(session, beforeEnum, afterEnum);
        if (err != MNN::NO_ERROR) {
            printf("FAIL: runSessionWithCallBackInfo error=%d\n", err);
            return 1;
        }
    }

    // Timed runs: measure every before→after pair using a stack
    int nRuns = 3;
    for (int run = 0; run < nRuns; run++) {
        std::stack<std::chrono::high_resolution_clock::time_point> opStartStack;
        int opIdx = 0;

        auto beforeTimed = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
            if (info) {
                opStartStack.push(std::chrono::high_resolution_clock::now());
            }
            return true;
        };
        auto afterTimed = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) -> bool {
            if (info && !opStartStack.empty()) {
                auto t1 = std::chrono::high_resolution_clock::now();
                auto t0 = opStartStack.top(); opStartStack.pop();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (opIdx < (int)opTimings.size()) {
                    opTimings[opIdx].ms += ms;
                    opTimings[opIdx].count++;
                }
                opIdx++;
            }
            return true;
        };

        auto t0 = std::chrono::high_resolution_clock::now();
        auto err = interpreter->runSessionWithCallBackInfo(session, beforeTimed, afterTimed);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err != MNN::NO_ERROR) {
            printf("  run %d: FAIL error=%d\n", run, err);
            break;
        }
        double runMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  run %d: %.2f ms total, %zu ops\n", run, runMs, opTimings.size());
    }

    // Print per-op breakdown (averaged over measured runs)
    printf("\n  %-40s %-18s %8s %10s %6s\n", "Op Name", "Type", "FLOPS(M)", "Time(ms)", "%");
    printf("  %s\n", std::string(80, '-').c_str());
    double sumMs = 0;
    for (auto& op : opTimings) {
        double avgOpMs = op.count > 0 ? op.ms / op.count : 0;
        sumMs += avgOpMs;
    }
    for (auto& op : opTimings) {
        double avgOpMs = op.count > 0 ? op.ms / op.count : 0;
        double pct = sumMs > 0 ? 100.0 * avgOpMs / sumMs : 0;
        printf("  %-40s %-18s %8.2f %10.3f %5.1f%%\n",
               op.name.c_str(), op.type.c_str(), op.flops, avgOpMs, pct);
    }
    printf("  %s\n", std::string(80, '-').c_str());
    printf("  %-40s %-18s %8s %10.3f\n", "TOTAL", "", "", sumMs);

    // ── 4. Session not stuck ──
    printf("\n=== SESSION RECOVERY CHECK ===\n");
    auto err = interpreter->runSession(session);
    printf("  %s\n", err == MNN::NO_ERROR ? "OK" : "FAIL");

    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    printf("\nDone.\n");
    return 0;
}
