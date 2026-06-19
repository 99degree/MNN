// // MNNProfiling.cpp
// MNN Profiling C API Implementation
//
// Created for profiling support
// Copyright © 2024, MNN Contributors
//

#include <MNN/Interpreter.hpp>
#include <MNN/MNNProfiling.h>

MNN_PUBLIC bool MNN_GetSessionInfo(void* interpreter, void* session, MNNSessionInfoCode code, void* ptr) {
    if (!interpreter || !session || !ptr) {
        return false;
    }
    
    auto* interp = static_cast<MNN::Interpreter*>(interpreter);
    auto* sess = static_cast<MNN::Session*>(session);
    
    return interp->getSessionInfo(sess, static_cast<MNN::Interpreter::SessionInfoCode>(code), ptr);
}

MNN_PUBLIC const char* MNN_GetVersion() {
    return MNN::getVersion();
}

MNN_PUBLIC float MNN_GetSessionMemory(void* interpreter, void* session) {
    float memory = -1.0f;
    if (MNN_GetSessionInfo(interpreter, session, MNN_SESSION_INFO_MEMORY, &memory)) {
        return memory;
    }
    return -1.0f;
}

MNN_PUBLIC float MNN_GetSessionFlops(void* interpreter, void* session) {
    float flops = -1.0f;
    if (MNN_GetSessionInfo(interpreter, session, MNN_SESSION_INFO_FLOPS, &flops)) {
        return flops;
    }
    return -1.0f;
}

MNN_PUBLIC int MNN_GetSessionThreadNumber(void* interpreter, void* session) {
    int threadNum = -1;
    if (MNN_GetSessionInfo(interpreter, session, MNN_SESSION_INFO_THREAD_NUMBER, &threadNum)) {
        return threadNum;
    }
    return -1;
}

MNN_PUBLIC int MNN_GetSessionBackends(void* interpreter, void* session, int* backends, int maxBackends) {
    if (!backends || maxBackends <= 0) {
        return -1;
    }
    
    // Try to get backend info
    std::vector<int32_t> tempBackends(maxBackends);
    if (MNN_GetSessionInfo(interpreter, session, MNN_SESSION_INFO_BACKENDS, tempBackends.data())) {
        // Copy to output array
        int count = 0;
        for (int i = 0; i < maxBackends && tempBackends[i] != 0; ++i) {
            backends[i] = tempBackends[i];
            count++;
        }
        return count;
    }
    return -1;
}