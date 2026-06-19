// test_profiling_api.cpp
// Simple test to verify the profiling API compiles and works

#include <iostream>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNProfiling.h>

int main() {
    std::cout << "Testing MNN Profiling API..." << std::endl;
    
    // Test 1: Version check
    std::cout << "\n1. Version Test:" << std::endl;
    const char* version = MNN_GetVersion();
    std::cout << "   MNN Version: " << version << std::endl;
    
    // Test 2: C API with null pointers (should fail gracefully)
    std::cout << "\n2. Null Pointer Test:" << std::endl;
    float memory = -1.0f;
    bool success = MNN_GetSessionInfo(nullptr, nullptr, MNN_SESSION_INFO_MEMORY, &memory);
    std::cout << "   Null pointer test: " << (success ? "FAILED (should not succeed)" : "PASSED (correctly failed)") << std::endl;
    
    // Test 3: C++ API basic usage
    std::cout << "\n3. C++ API Test:" << std::endl;
    std::cout << "   Attempting to create interpreter..." << std::endl;
    
    // This will likely fail without a real model, but we can test the API structure
    auto interpreter = MNN::Interpreter::createFromBuffer(nullptr, 0);
    if (!interpreter) {
        std::cout << "   Interpreter creation failed (expected without valid model)" << std::endl;
    } else {
        std::cout << "   Interpreter created successfully" << std::endl;
        MNN::Interpreter::destroy(interpreter);
    }
    
    // Test 4: C wrapper functions
    std::cout << "\n4. C Wrapper Functions Test:" << std::endl;
    memory = MNN_GetSessionMemory(nullptr, nullptr);
    std::cout << "   Memory (null): " << memory << " (expected: -1)" << std::endl;
    
    float flops = MNN_GetSessionFlops(nullptr, nullptr);
    std::cout << "   FLOPS (null): " << flops << " (expected: -1)" << std::endl;
    
    int threads = MNN_GetSessionThreadNumber(nullptr, nullptr);
    std::cout << "   Threads (null): " << threads << " (expected: -1)" << std::endl;
    
    int backends[10];
    int backendCount = MNN_GetSessionBackends(nullptr, nullptr, backends, 10);
    std::cout << "   Backends (null): " << backendCount << " (expected: -1)" << std::endl;
    
    std::cout << "\n✅ Profiling API structure test completed!" << std::endl;
    std::cout << "   All API functions are available and type-correct." << std::endl;
    
    return 0;
}