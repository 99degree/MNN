// example_profiling.cpp
// Example demonstrating MNN profiling API usage
//
// Build with: g++ example_profiling.cpp -I./install_profile/include -L./install_profile/lib -lMNN -o example_profiling

#include <iostream>
#include <vector>
#include <chrono>

// C API (if you need C compatibility)
#include <MNN/MNNProfiling.h>

// C++ API (recommended)
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>

// Function to demonstrate C API usage
void demonstrateCAPI() {
    std::cout << "=== MNN Profiling C API Demo ===" << std::endl;
    
    // In a real application, you would create these pointers from your model
    // For demonstration, we'll show the API usage
    void* interpreter = nullptr; // Would be Interpreter::createFromFile()
    void* session = nullptr;     // Would be interpreter->createSession()
    
    // Get MNN version
    const char* version = MNN_GetVersion();
    std::cout << "MNN Version: " << version << std::endl;
    
    // Example of getting memory usage
    float memory = MNN_GetSessionMemory(interpreter, session);
    std::cout << "Session Memory: " << memory << " MB" << std::endl;
    
    // Example of getting FLOPS
    float flops = MNN_GetSessionFlops(interpreter, session);
    std::cout << "Session FLOPS: " << flops << " M" << std::endl;
    
    // Example of getting thread count
    int threads = MNN_GetSessionThreadNumber(interpreter, session);
    std::cout << "Thread Count: " << threads << std::endl;
    
    // Example of getting backend info
    int backends[10];
    int backendCount = MNN_GetSessionBackends(interpreter, session, backends, 10);
    std::cout << "Backends (" << backendCount << "): ";
    for (int i = 0; i < backendCount; ++i) {
        std::cout << backends[i] << " ";
    }
    std::cout << std::endl;
    
    // Direct getSessionInfo usage
    float memoryDirect = 0;
    bool success = MNN_GetSessionInfo(interpreter, session, MNN_SESSION_INFO_MEMORY, &memoryDirect);
    if (success) {
        std::cout << "Direct Memory: " << memoryDirect << " MB" << std::endl;
    }
    
    std::cout << std::endl;
}

// Function to demonstrate C++ API usage (recommended)
void demonstrateCPPAPI(const std::string& modelPath) {
    std::cout << "=== MNN Profiling C++ API Demo ===" << std::endl;
    
    // Create interpreter
    auto interpreter = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (!interpreter) {
        std::cerr << "Failed to create interpreter from: " << modelPath << std::endl;
        return;
    }
    
    std::cout << "Model loaded successfully" << std::endl;
    
    // Configure session
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;  // Use CPU backend
    config.numThread = 4;           // Use 4 threads
    
    // Create session
    auto session = interpreter->createSession(config);
    if (!session) {
        std::cerr << "Failed to create session" << std::endl;
        MNN::Interpreter::destroy(interpreter);
        return;
    }
    
    // Profile before execution
    std::cout << "\n--- Pre-execution Profile ---" << std::endl;
    
    float memoryBefore = 0;
    if (interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryBefore)) {
        std::cout << "Memory usage: " << memoryBefore << " MB" << std::endl;
    }
    
    float flops = 0;
    if (interpreter->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops)) {
        std::cout << "FLOPS: " << flops << " M" << std::endl;
    }
    
    int threadCount = 0;
    if (interpreter->getSessionInfo(session, MNN::Interpreter::THREAD_NUMBER, &threadCount)) {
        std::cout << "Thread count: " << threadCount << std::endl;
    }
    
    // Get backend information
    std::vector<int32_t> backends(10);
    if (interpreter->getSessionInfo(session, MNN::Interpreter::BACKENDS, backends.data())) {
        std::cout << "Backends: ";
        for (int i = 0; i < 10 && backends[i] != 0; ++i) {
            std::cout << backends[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Get input tensor information
    auto inputTensor = interpreter->getSessionInput(session, nullptr);
    if (inputTensor) {
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < inputTensor->dimensions(); ++i) {
            std::cout << inputTensor->length(i) << " ";
        }
        std::cout << std::endl;
    }
    
    // Run inference (with timing)
    std::cout << "\n--- Running Inference ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto error = interpreter->runSession(session);
    if (error != MNN::NO_ERROR) {
        std::cerr << "Inference failed with error: " << error << std::endl;
    } else {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    }
    
    // Profile after execution
    std::cout << "\n--- Post-execution Profile ---" << std::endl;
    
    float memoryAfter = 0;
    if (interpreter->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryAfter)) {
        std::cout << "Memory usage: " << memoryAfter << " MB" << std::endl;
        std::cout << "Memory delta: " << (memoryAfter - memoryBefore) << " MB" << std::endl;
    }
    
    // Get resize status
    int resizeStatus = 0;
    if (interpreter->getSessionInfo(session, MNN::Interpreter::RESIZE_STATUS, &resizeStatus)) {
        std::cout << "Resize status: " << resizeStatus << " (0=ready, 1=need_malloc, 2=need_resize)" << std::endl;
    }
    
    // Get output tensor information
    auto outputTensor = interpreter->getSessionOutput(session, nullptr);
    if (outputTensor) {
        std::cout << "Output tensor shape: ";
        for (int i = 0; i < outputTensor->dimensions(); ++i) {
            std::cout << outputTensor->length(i) << " ";
        }
        std::cout << std::endl;
    }
    
    // Clean up
    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
    
    std::cout << "\nSession cleaned up" << std::endl;
}

// Advanced: Callback-based profiling
void demonstrateCallbackProfiling(const std::string& modelPath) {
    std::cout << "=== MNN Profiling with Callbacks ===" << std::endl;
    
    auto interpreter = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (!interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return;
    }
    
    // Enable debug session to allow callbacks
    interpreter->setSessionMode(MNN::Interpreter::Session_Debug);
    
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 2;
    
    auto session = interpreter->createSession(config);
    if (!session) {
        std::cerr << "Failed to create session" << std::endl;
        MNN::Interpreter::destroy(interpreter);
        return;
    }
    
    // Define callbacks for detailed profiling
    MNN::TensorCallBackWithInfo beforeCallback = [](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        std::cout << "Before op: " << info->name() << " (type: " << info->type() << ", flops: " << info->flops() << " M)" << std::endl;
        return true; // Continue execution
    };
    
    MNN::TensorCallBackWithInfo afterCallback = [](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        std::cout << "After op: " << info->name() << std::endl;
        return true; // Continue execution
    };
    
    std::cout << "Running with callbacks..." << std::endl;
    auto error = interpreter->runSessionWithCallBackInfo(session, beforeCallback, afterCallback, true);
    
    if (error != MNN::NO_ERROR) {
        std::cerr << "Callback inference failed: " << error << std::endl;
    }
    
    // Clean up
    interpreter->releaseSession(session);
    MNN::Interpreter::destroy(interpreter);
}

int main(int argc, char* argv[]) {
    std::string modelPath = "model.mnn"; // Default model path
    
    if (argc > 1) {
        modelPath = argv[1];
    }
    
    // Show available options
    std::cout << "MNN Profiling Example" << std::endl;
    std::cout << "Usage: " << argv[0] << " [model.mnn]" << std::endl;
    std::cout << std::endl;
    
    // Demonstrate C API
    demonstrateCAPI();
    
    // Demonstrate C++ API
    demonstrateCPPAPI(modelPath);
    
    // Demonstrate callback-based profiling
    demonstrateCallbackProfiling(modelPath);
    
    return 0;
}