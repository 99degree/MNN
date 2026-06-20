#include <stdio.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
int main() {
    auto* interp = MNN::Interpreter::createFromFile("/data/data/com.termux/files/home/softisp/cam-rust/profile_4k.mnn");
    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    auto* session = interp->createSession(cfg);

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
    interp->runSession(session);
    printf("run OK\n"); fflush(stdout);

    auto* disp = interp->getSessionOutput(session, "DisplayBlock/frame");
    if (!disp) { printf("FAIL: no output\n"); return 1; }
    printf("DisplayBlock/frame: [%d,%d,%d,%d] type=%d bits=%d\n",
        disp->shape()[0], disp->shape()[1],
        disp->shape()[2], disp->shape()[3],
        disp->getType().code, disp->getType().bits);

    // Read via tensor host pointer (CPU backend always has host)
    auto* host_ptr = disp->host<uint8_t>();
    if (host_ptr) {
        // Read first 4 bytes as uint32
        uint32_t first;
        memcpy(&first, host_ptr, 4);
        printf("First pixel: 0x%08X  R=%d G=%d B=%d A=%d\n",
            first, (first>>24)&0xFF, (first>>16)&0xFF, (first>>8)&0xFF, first&0xFF);
        
        // Show several pixels
        printf("First 4 pixels:\n");
        for (int i = 0; i < 4 && i < disp->shape()[3]; i++) {
            memcpy(&first, host_ptr + i * 4, 4);
            printf("  [%d] 0x%08X R=%d G=%d B=%d A=%d\n", i,
                first, (first>>24)&0xFF, (first>>16)&0xFF, (first>>8)&0xFF, first&0xFF);
        }
    } else {
        printf("host pointer is null\n");
    }
    fflush(stdout);

    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    return 0;
}
