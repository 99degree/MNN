// // MNNProfiling.h
// MNN Profiling C API
//
// Created for profiling support
// Copyright © 2024, MNN Contributors
//
#ifndef MNN_Profiling_h
#define MNN_Profiling_h

#include <MNN/MNNDefine.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Session information codes for profiling
 */
typedef enum {
    MNN_SESSION_INFO_MEMORY = 0,     ///< Memory usage in MB (float*)
    MNN_SESSION_INFO_FLOPS = 1,       ///< Floating point operations in M (float*)
    MNN_SESSION_INFO_BACKENDS = 2,   ///< Backend types used (int*)
    MNN_SESSION_INFO_RESIZE_STATUS = 3, ///< Resize status (int*)
    MNN_SESSION_INFO_THREAD_NUMBER = 4, ///< Number of threads (int*)
} MNNSessionInfoCode;

/**
 * @brief Get session profiling information
 * @param interpreter Pointer to MNN Interpreter instance
 * @param session Pointer to MNN Session instance
 * @param code Information code to retrieve
 * @param ptr Pointer to store the result (type depends on code)
 * @return true if successful, false otherwise
 * @note This is a C wrapper around Interpreter::getSessionInfo()
 */
MNN_PUBLIC bool MNN_GetSessionInfo(void* interpreter, void* session, MNNSessionInfoCode code, void* ptr);

/**
 * @brief Get MNN version string
 * @return Version string (e.g., "3.6.0")
 */
MNN_PUBLIC const char* MNN_GetVersion();

/**
 * @brief Get session memory usage in MB
 * @param interpreter Pointer to MNN Interpreter instance
 * @param session Pointer to MNN Session instance
 * @return Memory usage in MB, or -1.0f on error
 */
MNN_PUBLIC float MNN_GetSessionMemory(void* interpreter, void* session);

/**
 * @brief Get session FLOPS in M
 * @param interpreter Pointer to MNN Interpreter instance
 * @param session Pointer to MNN Session instance
 * @return FLOPS in M, or -1.0f on error
 */
MNN_PUBLIC float MNN_GetSessionFlops(void* interpreter, void* session);

/**
 * @brief Get session thread count
 * @param interpreter Pointer to MNN Interpreter instance
 * @param session Pointer to MNN Session instance
 * @return Thread count, or -1 on error
 */
MNN_PUBLIC int MNN_GetSessionThreadNumber(void* interpreter, void* session);

/**
 * @brief Get backend types used by session
 * @param interpreter Pointer to MNN Interpreter instance
 * @param session Pointer to MNN Session instance
 * @param backends Array to store backend types (must be pre-allocated)
 * @param maxBackends Maximum number of backends to store
 * @return Number of backends stored, or -1 on error
 */
MNN_PUBLIC int MNN_GetSessionBackends(void* interpreter, void* session, int* backends, int maxBackends);

#ifdef __cplusplus
}
#endif

#endif /* MNN_Profiling_h */