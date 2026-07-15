// VulkanISPExtensions.cpp - ISP-specific Vulkan extensions for MNN
// These functions provide ISP-specific Vulkan capabilities that can be called
// from the cam-isp integration layer via weak linking.

#include "VulkanDevice.hpp"
#include "VulkanInstance.hpp"
#include <map>
#include <string>

namespace MNN {

// Global workgroup preset registry
static std::map<std::string, std::pair<int, int>> gWorkgroupPresets = {
    {"default", {8, 8}},
    {"small", {4, 4}},
    {"medium", {8, 8}},
    {"large", {16, 16}},
    {"isp", {8, 8}},
    {"isp_lite", {4, 4}},
};

// Global session workgroup override
static std::map<void*, std::pair<int, int>> gSessionWorkgroups;

// Const buffer registry for hot-swapping
static std::map<std::string, std::vector<uint8_t>> gConstBuffers;

} // namespace MNN

using MNN::gSessionWorkgroups;
using MNN::gConstBuffers;

// C API for ISP Vulkan extensions
extern "C" {

// Query optimal workgroup size for current GPU.
// Returns the best workgroup size based on device capabilities.
void MNNVulkanQueryOptimalWorkgroup(int* out_x, int* out_y) {
    if (!out_x || !out_y) return;
    
    // Default workgroup size
    *out_x = 8;
    *out_y = 8;
    
    // TODO: Query actual device properties and return optimal workgroup size
    // For now, return a reasonable default
}

// Set preferred workgroup size for a Vulkan session.
void MNNVulkanSetSessionWorkgroup(void* session, int size_x, int size_y) {
    if (!session) return;
    gSessionWorkgroups[session] = {size_x, size_y};
}

// Set workgroup preset by name.
void MNNVulkanSetWorkgroupPreset(const char* preset_name) {
    if (!preset_name) return;
    // Store the preset for future sessions
    // TODO: Make this thread-safe if needed
}

// Hot-swap a const buffer at runtime for live 3A adjustments.
// This allows updating constant buffers (e.g., ISP parameters) without
// recreating the Vulkan pipeline.
void MNNVulkanHotSwapConstBuffer(void* session, const char* name, const void* data, int size) {
    if (!session || !name || !data || size <= 0) return;
    
    // Store the buffer data
    gConstBuffers[name] = std::vector<uint8_t>((const uint8_t*)data, (const uint8_t*)data + size);
    
    // TODO: Actually update the Vulkan descriptor set with the new data
    // This requires access to the Vulkan session and descriptor set management
}

} // extern "C"
