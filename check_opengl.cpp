#include <stdio.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

int main() {
    auto* interp = MNN::Interpreter::createFromFile("/data/data/com.termux/files/home/softisp/cam-rust/profile_hd.mnn");
    if (!interp) { printf("FAIL: load model\n"); return 1; }

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_OPENGL;
    cfg.numThread = 4;

    interp->setSessionMode(MNN::Interpreter::Session_Release);
    auto* session = interp->createSession(cfg);
    if (session) {
        printf("OpenGL session created OK\n");
        interp->releaseSession(session);
    } else {
        printf("OpenGL session FAILED\n");
        
        // Try with Session_Debug
        interp->setSessionMode(MNN::Interpreter::Session_Debug);
        session = interp->createSession(cfg);
        if (session) {
            printf("OpenGL session (Debug) created OK\n");
            interp->releaseSession(session);
        } else {
            printf("OpenGL session (Debug) also FAILED\n");
        }
    }

    MNN::Interpreter::destroy(interp);
    return 0;
}
