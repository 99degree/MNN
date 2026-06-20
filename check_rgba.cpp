#include <stdio.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
int main() {
    auto* interp = MNN::Interpreter::createFromFile("/data/data/com.termux/files/home/softisp/cam-rust/profile_4k.mnn");
    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    auto* session = interp->createSession(cfg);
    if (!session) { printf("FAIL\n"); return 1; }
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

    auto* disp = interp->getSessionOutput(session, "DisplayBlock/frame");
    if (disp) {
        printf("DisplayBlock/frame: [%d,%d,%d,%d] type=%d bits=%d\n",
            disp->shape()[0], disp->shape()[1],
            disp->shape()[2], disp->shape()[3],
            disp->getType().code, disp->getType().bits);
        // Verify first pixel
        if (disp->host<int32_t>()) {
            uint32_t first = *(uint32_t*)disp->host<int32_t>();
            printf("First pixel: 0x%08X\n", first);
            printf("  R=%d G=%d B=%d A=%d\n",
                (first >> 24) & 0xFF, (first >> 16) & 0xFF,
                (first >> 8) & 0xFF, first & 0xFF);
        }
    }
    // Check aux output for FHD dims
    auto* aux = interp->getSessionOutput(session, "aux_hook_src/out");
    if (aux) {
        printf("aux_hook_src/out: [%d,%d,%d,%d]\n",
            aux->shape()[0], aux->shape()[1], aux->shape()[2], aux->shape()[3]);
    }
    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    return 0;
}
