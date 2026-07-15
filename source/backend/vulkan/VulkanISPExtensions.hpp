// VulkanISPExtensions.hpp - ISP-specific Vulkan extensions for MNN
// These functions provide ISP-specific Vulkan capabilities that can be called
// from the cam-isp integration layer via weak linking.

#ifndef MNN_VulkanISPExtensions_hpp
#define MNN_VulkanISPExtensions_hpp

#ifdef __cplusplus
extern "C" {
#endif

// Query optimal workgroup size for current GPU.
// Returns the best workgroup size based on device capabilities.
void MNNVulkanQueryOptimalWorkgroup(int* out_x, int* out_y);

// Set preferred workgroup size for a Vulkan session.
void MNNVulkanSetSessionWorkgroup(void* session, int size_x, int size_y);

// Set workgroup preset by name.
void MNNVulkanSetWorkgroupPreset(const char* preset_name);

// Hot-swap a const buffer at runtime for live 3A adjustments.
// This allows updating constant buffers (e.g., ISP parameters) without
// recreating the Vulkan pipeline.
void MNNVulkanHotSwapConstBuffer(void* session, const char* name, const void* data, int size);

#ifdef __cplusplus
}
#endif

#endif /* MNN_VulkanISPExtensions_hpp */
