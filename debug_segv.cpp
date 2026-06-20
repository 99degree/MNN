#include <stdio.h>
#include <signal.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

int main() {
    fprintf(stderr, "A\n");
    auto* interp = MNN::Interpreter::createFromFile("/data/data/com.termux/files/home/softisp/cam-rust/profile_4k.mnn");
    if (!interp) { fprintf(stderr, "FAIL: load\n"); return 1; }
    fprintf(stderr, "B\n");
    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    auto* session = interp->createSession(cfg);
    if (!session) { fprintf(stderr, "FAIL: session\n"); return 1; }
    fprintf(stderr, "C\n");
    auto* input = interp->getSessionInput(session, nullptr);
    fprintf(stderr, "D input=%p\n", (void*)input);
    if (input && input->host<int32_t>()) {
        fprintf(stderr, "D1\n");
        auto* d = input->host<int32_t>();
        for (size_t j = 0; j < input->elementSize(); j++) d[j] = (int32_t)(j % 65536);
    }
    fprintf(stderr, "E\n");
    const char* extras[] = {"zzz_FcsBlock/fcs_gain_scaled","zzz_LdciBlock/ldci_strength_scaled","zzz_EeBlock/ee_gain_scaled"};
    for (auto* en : extras) {
        auto* t = interp->getSessionInput(session, en);
        if (t && t->host<float>()) { auto* d = t->host<float>(); for (size_t j = 0; j < t->elementSize(); j++) d[j] = 0.1f; }
    }
    fprintf(stderr, "F\n");
    interp->resizeSession(session);
    fprintf(stderr, "G\n");
    auto err = interp->runSession(session);
    fprintf(stderr, "H err=%d\n", (int)err);
    auto* disp = interp->getSessionOutput(session, "DisplayBlock/frame");
    fprintf(stderr, "I disp=%p\n", (void*)disp);
    if (disp) {
        fprintf(stderr, "Shape: [%d,%d,%d,%d]\n",
            disp->shape()[0], disp->shape()[1], disp->shape()[2], disp->shape()[3]);
        auto* ptr = disp->host<uint8_t>();
        fprintf(stderr, "J ptr=%p\n", (void*)ptr);
    }
    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    fprintf(stderr, "OK\n");
    return 0;
}
