#include <stdio.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
int main() {
    auto* interp = MNN::Interpreter::createFromFile("/data/data/com.termux/files/home/softisp/cam-rust/profile_4k.mnn");
    if (!interp) { printf("FAIL: load\n"); return 1; }

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
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
    interp->runSession(session);
    printf("run OK\n"); fflush(stdout);

    // Check DisplayBlock/frame
    auto* disp = interp->getSessionOutput(session, "DisplayBlock/frame");
    if (disp) {
        printf("DisplayBlock/frame: [%d,%d,%d,%d] type=%d bits=%d\n",
            disp->shape()[0], disp->shape()[1],
            disp->shape()[2], disp->shape()[3],
            disp->getType().code, disp->getType().bits);
        fflush(stdout);

        // Copy tensor to host if needed
        auto host = disp->host<int32_t>();
        if (host) {
            uint32_t first = *(const uint32_t*)host;
            printf("First pixel: 0x%08X  R=%d G=%d B=%d A=%d\n",
                first, (first>>24)&0xFF, (first>>16)&0xFF, (first>>8)&0xFF, first&0xFF);
        } else {
            // Try creating a host tensor and copy
            printf("host pointer null, creating host copy...\n"); fflush(stdout);
            auto host_tensor = new MNN::Tensor(disp, MNN::Tensor::CAFFE);
            disp->copyToHostTensor(host_tensor);
            if (host_tensor->host<int32_t>()) {
                uint32_t val = *(const uint32_t*)host_tensor->host<int32_t>();
                printf("First pixel (copied): 0x%08X  R=%d G=%d B=%d A=%d\n",
                    val, (val>>24)&0xFF, (val>>16)&0xFF, (val>>8)&0xFF, val&0xFF);
            }
            delete host_tensor;
        }
    } else {
        printf("DisplayBlock/frame: null\n");
    }
    fflush(stdout);

    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    return 0;
}
