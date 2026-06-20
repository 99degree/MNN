#include <stdio.h>
#include <MNN/Interpreter.hpp>
int main() {
    printf("Loading model...\n"); fflush(stdout);
    auto* interp = MNN::Interpreter::createFromFile("/data/data/com.termux/files/home/softisp/cam-rust/profile_4k.mnn");
    if (!interp) { printf("FAIL: load\n"); return 1; }
    printf("Creating session...\n"); fflush(stdout);
    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    auto* session = interp->createSession(cfg);
    if (!session) { printf("FAIL: session\n"); return 1; }
    printf("Session OK\n"); fflush(stdout);
    auto* input = interp->getSessionInput(session, nullptr);
    if (!input) { printf("FAIL: no input\n"); return 1; }
    printf("Input: [%d,%d,%d,%d]\n",
        input->shape()[0], input->shape()[1],
        input->shape()[2], input->shape()[3]);
    interp->resizeSession(session);
    printf("Resize OK\n"); fflush(stdout);
    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    printf("OK\n");
    return 0;
}
