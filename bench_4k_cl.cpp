#include <stdio.h>
#include <chrono>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
int main() {
    auto* interp = MNN::Interpreter::createFromFile("/data/data/com.termux/files/home/softisp/cam-rust/profile_4k.mnn");
    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_OPENCL;
    cfg.numThread = 4;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    cfg.backendConfig = &bc;
    interp->setSessionMode(MNN::Interpreter::Session_Debug);

    auto* session = interp->createSession(cfg);
    if (!session) { printf("FAIL: session\n"); return 1; }

    auto* input = interp->getSessionInput(session, nullptr);
    if (input && input->host<int32_t>()) {
        auto* d = input->host<int32_t>();
        for (size_t j = 0; j < input->elementSize(); j++) d[j] = (int32_t)(j % 65536);
    }
    const char* extras[] = {"zzz_FcsBlock/fcs_gain_scaled","zzz_LdciBlock/ldci_strength_scaled","zzz_EeBlock/ee_gain_scaled"};
    for (auto* en : extras) {
        auto* t = interp->getSessionInput(session, en);
        if (t && t->host<float>()) { auto* d = t->host<float>(); for (size_t j = 0; j < t->elementSize(); j++) d[j] = 0.1f; }
    }
    interp->resizeSession(session);
    interp->runSession(session); // warmup

    printf("Input: [1,1,2160,3840] uint16 Bayer\n\n");

    // Read the actual DISPLAY output (PackedRgb: [1,1,1080,1920] INT32)
    auto* display = interp->getSessionOutput(session, "DisplayBlock/frame");
    if (display) {
        printf("DisplayBlock/frame: [%d,%d,%d,%d] type=%d bits=%d — INT32 packed RGBA\n",
            display->shape()[0], display->shape()[1],
            display->shape()[2], display->shape()[3],
            display->getType().code, display->getType().bits);
    }

    int iters = 3;
    double total = 0;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        interp->runSession(session);
        auto t1 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(t1-t0).count();
    }
    printf("OpenCL 4K Bayer→FHD Packed: %.2f ms (%d iters)\n", total/iters, iters);

    float mem = 0, flops = 0;
    interp->getSessionInfo(session, MNN::Interpreter::MEMORY, &mem);
    interp->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    printf("MEM: %.2f MB  FLOPS: %.2f M\n", mem, flops);

    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    return 0;
}
