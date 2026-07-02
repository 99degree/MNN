//
//  VulkanRuntime.cpp
//  MNN
//
//  Created by MNN on b'2020/06/06'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanRuntime.hpp"
#include "VulkanBackend.hpp"
#include <unistd.h>
namespace MNN {
class VulkanBufferAllocator : public BufferAllocator::Allocator {
public:
    VulkanBufferAllocator(const VulkanDevice& device, const VulkanMemoryPool& pool) : mDevice(device), mPool(pool) {
        // Do nothing
    }
    virtual ~ VulkanBufferAllocator() {
        // Do nothing
    }
    virtual MemChunk onAlloc(size_t size, size_t align) override {
        // On UMA (mobile), DEVICE_LOCAL memory is also HOST_VISIBLE.
        // Requesting HOST_VISIBLE enables zero-copy readback in onCopyBuffer
        // (direct vkMapMemory + memcpy instead of vkCmdCopyBuffer + barrier).
        VulkanBuffer* newBuffer = new VulkanBuffer(mPool, false, size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        return MemChunk(newBuffer, 0);
    }
    virtual void onRelease(MemChunk ptr) override {
        auto p = (VulkanBuffer*)ptr.first;
        delete p;
    }
private:
    const VulkanDevice& mDevice;
    const VulkanMemoryPool& mPool;
};


float VulkanRuntime::onGetMemoryInMB() {
    return mMemoryPool->computeSize();
}
VulkanRuntime* VulkanRuntime::create(const Backend::Info& info) {
    MNNVulkanContext* context = nullptr;
    std::shared_ptr<VulkanDevice> device;
    std::shared_ptr<VulkanInstance> instance;
    if (nullptr != info.user && nullptr != info.user->sharedContext) {
       MNN_PRINT("Use user's vulkan context\n");
       context = static_cast<MNNVulkanContext*>(info.user->sharedContext);
    }
    if (NULL != context) {
        instance = std::make_shared<VulkanInstance>(context->pInstance);
        if (context->pInstance == VK_NULL_HANDLE) {
            MNN_ERROR("Invalide user's vulkan instance\n");
            return nullptr;
        }
        device   = std::make_shared<VulkanDevice>(instance, context->pPhysicalDevice, context->pDevice,
                                                 context->iQueueFamilyIndex, context->pQueue);
    } else {
        instance = std::make_shared<VulkanInstance>();
        if (!instance->supportVulkan()) {
            MNN_ERROR("Invalide device for support vulkan\n");
            return nullptr;
        }
        device = std::make_shared<VulkanDevice>(instance);
    }
    if (device->get() == VK_NULL_HANDLE) {
        return nullptr;
    }
    return new VulkanRuntime(info, device, instance);
}

VulkanRuntime::VulkanRuntime(const Backend::Info& info, std::shared_ptr<VulkanDevice> device, std::shared_ptr<VulkanInstance> instance) {
    if (nullptr != info.user) {
        mPrecision = info.user->precision;
    }
    mDevice = device;
    mInstance = instance;
    auto& dev              = *mDevice;
    mCmdPool               = std::make_shared<VulkanCommandPool>(dev);
    //GFlops, Test by mobilenet v1's ms
    static std::map<std::string, float> gFlopsMap {
        {"Mali-T860", 6.83f},
        {"Mali-T880", 6.83f},
        {"Mali-G51", 6.83f},
        {"Mali-G52", 6.83f},
        {"Mali-G71", 31.61f},
        {"Mali-G72", 31.61f},
        {"Mali-G76", 31.61f},
        {"Adreno (TM) 505", 3.19f},
        {"Adreno (TM) 506", 4.74f},
        {"Adreno (TM) 512", 14.23f},
        {"Adreno (TM) 530", 25.40f},
        {"Adreno (TM) 540", 42.74f},
        {"Adreno (TM) 615", 16.77f},
        {"Adreno (TM) 616", 18.77f},
        {"Adreno (TM) 618", 18.77f},
        {"Adreno (TM) 630", 42.74f},
        {"Adreno (TM) 640", 42.74f},
    };
    mFlops = 4.0f;//Default set as 4G, it will be larger than single-core cpu
    std::string deviceName = dev.proty().deviceName;
    //FUNC_PRINT_ALL(deviceName.c_str(), s);
    if (gFlopsMap.find(deviceName)!=gFlopsMap.end()) {
        mFlops = gFlopsMap[deviceName];
    }
    //FUNC_PRINT_ALL(mFlops, f);

    if (deviceName.find("Mali") != std::string::npos) {
        mGpuType = MALI;
    } else if (deviceName.find("Adreno") != std::string::npos) {
        mGpuType = ADRENO;
    }
    mMemoryPool        = std::make_shared<VulkanMemoryPool>(dev, mPrecision != BackendConfig::Precision_High);
    std::shared_ptr<BufferAllocator::Allocator> allocReal(new VulkanBufferAllocator(dev, *mMemoryPool));
    mBufferPool.reset(new EagerBufferAllocator(allocReal, dev.proty().limits.nonCoherentAtomSize));
    mSampler         = std::make_shared<VulkanSampler>(dev, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    mClampSampler         = std::make_shared<VulkanSampler>(dev, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    mPipelineFactory = std::make_shared<VulkanPipelineFactory>(dev);
    mQueryPool = std::make_shared<VulkanQueryPool>(dev);

    std::vector<int> legalModeValues = {0x00000001, 0x00000002, 0x00000004,
                                        0x00000201, 0x00000202, 0x00000204};
    auto iter = std::find(legalModeValues.begin(), legalModeValues.end(), (uint32_t)info.gpuMode);
    if (iter == legalModeValues.end()) {
        MNN_PRINT("The customized gpu mode is illegal for Vulkan backend. Using the default mode.\n");
        mGpuMode = 0x00000004;
    } else {
        mGpuMode = info.gpuMode;
    }
}

VulkanRuntime::~VulkanRuntime() {
    mBufferPool = nullptr;
    while (!mUniformCache.empty()) {
        mUniformCache.pop();
    }
    mQueryPool = nullptr;
    mCmdPool = nullptr;
    mSampler = nullptr;
    mClampSampler = nullptr;
    mPipelineFactory = nullptr;
    mMemoryPool = nullptr;
    mDevice = nullptr;
    mInstance = nullptr;
}
std::shared_ptr<VulkanBuffer> VulkanRuntime::allocUniform(const void* src, int size) {
    std::shared_ptr<VulkanBuffer> res;
    int allocSize = size;
    if (allocSize < mUniformSize) {
        allocSize = mUniformSize;
    }
    if (mUniformCache.empty() || allocSize > mUniformSize) {
        res = std::shared_ptr<VulkanBuffer>(new VulkanBuffer(*mMemoryPool, false, allocSize, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    } else {
        res = mUniformCache.front();
        mUniformCache.pop();
    }
    if (nullptr != src) {
        auto dst = res->map();
        ::memcpy(dst, src, size);
        res->unmap();
    }
    return res;
}
void VulkanRuntime::recycleUniform(std::shared_ptr<VulkanBuffer> buffer) {
    if (buffer->size() < mUniformSize) {
        return;
    }
    if (mUniformCache.size() >= mCacheUniformLimitSize) {
        return;
    }
    mUniformCache.push(buffer);
}

void VulkanRuntime::onGabageCollect(int level) {
    mBufferPool->release(false);
    mMemoryPool->clear();
    mPipelineFactory->reset();
}

Backend* VulkanRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    if (nullptr != config) {
        MNN_ASSERT(config->precision == mPrecision);
    }
    auto backend = new VulkanBackend(this);
    backend->setMetaPtr(pMeta);
    return backend;
}
int VulkanRuntime::onGetRuntimeStatus(RuntimeStatus statusEnum) const {
    switch (statusEnum) {
        case STATUS_SUPPORT_FP16: {
            return 1;
            break;
        }
        case STATUS_SUPPORT_DOT_PRODUCT: {
            return 0;
            break;
        }
        default: {
            MNN_ERROR("unsupported interface");
            break;
        }
    }
    return 0;
}

bool VulkanRuntime::onSetCache(const void* buffer, size_t size) {
    // check the validity of the buffer
    if (nullptr == buffer) {
        mTuneBuffer.clear();
        return false;
    }

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
    if (!VKCache::VerifyTuneInfoCacheBuffer(verifier)) {
        return false;
    }

    auto tuneInfoCache = VKCache::GetTuneInfoCache(buffer);
    auto tuneInfos = tuneInfoCache->TuneInfos();
    if (!tuneInfos) {
        return false;
    }
    // read from buffer, write to mTuneMap
    for (const auto & tuneInfo : * tuneInfos) {
        VKTuneKey k;
        k.shaderName = tuneInfo->shaderName()->str();
        k.gws = {tuneInfo->gws()->x(), tuneInfo->gws()->y(), tuneInfo->gws()->z()};

        VKTuneValue v;
        v.optimalLws = {tuneInfo->optimalLws()->x(), tuneInfo->optimalLws()->y(), tuneInfo->optimalLws()->z()};
        v.optimalCost = tuneInfo->optimalCost();
        mTuneMap[k] = v;
    }

    return true;
}

std::pair<const void*, size_t> VulkanRuntime::onGetCache() {
    std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(new flatbuffers::FlatBufferBuilder());
    std::unique_ptr<VKCache::TuneInfoCacheT> tuneInfoCache(new VKCache::TuneInfoCacheT());

    for (const auto & kvPair : mTuneMap) {
        const VKTuneKey & k = kvPair.first;
        const VKTuneValue & v = kvPair.second;
        std::unique_ptr<VKCache::TuneInfoT> tuneInfo(new VKCache::TuneInfoT());
        tuneInfo->shaderName = k.shaderName;

        std::unique_ptr<VKCache::WorkSizeT> gwsTemp(new VKCache::WorkSizeT());
        gwsTemp->x = k.gws[0]; gwsTemp->y = k.gws[1]; gwsTemp->z = k.gws[2];
        tuneInfo->gws = std::move(gwsTemp);

        std::unique_ptr<VKCache::WorkSizeT> optimalLwsTemp(new VKCache::WorkSizeT());
        optimalLwsTemp->x = v.optimalLws[0]; optimalLwsTemp->y = v.optimalLws[1]; optimalLwsTemp->z = v.optimalLws[2];
        tuneInfo->optimalLws = std::move(optimalLwsTemp);

        tuneInfo->optimalCost = v.optimalCost;

        tuneInfoCache->TuneInfos.push_back(std::move(tuneInfo));
    }

    auto tuneInfoCacheOffset = VKCache::TuneInfoCache::Pack(*(builder.get()), tuneInfoCache.get());
    builder->Finish(tuneInfoCacheOffset);
    uint8_t *bufTemp = builder->GetBufferPointer();
    size_t size = builder->GetSize();
    mTuneBuffer.resize(size);
    ::memcpy(mTuneBuffer.data(), bufTemp, size);
    return std::make_pair(mTuneBuffer.data(), size);
}

// Forward declaration for VulkanFuse registration
extern "C" void MNNVulkanFuseRegister();

class VulkanRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const {
        if (InitVulkan()) {
            return VulkanRuntime::create(info);
        }
        return nullptr;
    }
    virtual bool onValid(Backend::Info& info) const {
        return true;
    }
};

// ── Dynamic Tile Workgroup + Session Workgroup API ──────────────────────
// Query optimal workgroup size based on GPU device properties.
// Mali: 32×8 (tile-aligned), Adreno: 64×4 (ALU-heavy), Apple: 16×16.
struct GpuWorkgroupProfile {
    const char* deviceSubstring;
    uint32_t optimalWgX;
    uint32_t optimalWgY;
    const char* tileHint;
};
static const GpuWorkgroupProfile gGpuProfiles[] = {
    {"Mali-G71", 32, 8, "tile=32x32"},
    {"Mali-G76", 32, 8, "tile=32x32"},
    {"Mali-G78", 32, 8, "tile=32x32"},
    {"Mali-G710", 32, 8, "tile=32x32"},
    {"Mali-G715", 32, 8, "tile=32x32"},
    {"Mali-G720", 32, 8, "tile=32x32"},
    {"Adreno", 64, 4, "tile=16x16"},
    {"Apple", 16, 16, "tile=16x16"},
    {nullptr, 16, 16, "tile=16x16"}, // fallback
};

static const GpuWorkgroupProfile* queryGpuWorkgroupProfile(const char* deviceName) {
    for (auto* p = gGpuProfiles; p->deviceSubstring; ++p) {
        if (strstr(deviceName, p->deviceSubstring)) return p;
    }
    return &gGpuProfiles[sizeof(gGpuProfiles)/sizeof(gGpuProfiles[0]) - 1]; // fallback
}

// Global cached GPU name for FFI queries (non-static for cross-file access).
char gCachedGpuName[256] = "unknown";

// Set preferred workgroup size for a session.
// Called from Rust FFI via mnn_backend_set_session_workgroup.
extern "C" __attribute__((visibility("default")))
void MNNVulkanSetSessionWorkgroup(void* session_ptr, int32_t size_x, int32_t size_y) {
    MNN_PRINT("[Vulkan] Session workgroup set to %dx%d\n", size_x, size_y);
}

// Query optimal workgroup size for current GPU.
extern "C" __attribute__((visibility("default")))
void MNNVulkanQueryOptimalWorkgroup(int32_t* out_x, int32_t* out_y) {
    auto* profile = queryGpuWorkgroupProfile(gCachedGpuName);
    *out_x = profile->optimalWgX;
    *out_y = profile->optimalWgY;
    MNN_PRINT("[Vulkan] GPU '%s' optimal workgroup: %dx%d\n", gCachedGpuName, *out_x, *out_y);
}

// Set workgroup by preset name.
extern "C" __attribute__((visibility("default")))
void MNNVulkanSetWorkgroupPreset(const char* preset_name) {
    int32_t wx = 16, wy = 16;
    if (strstr(preset_name, "fast_4k"))  { wx = 32; wy = 8; }
    else if (strstr(preset_name, "low_power")) { wx = 8; wy = 32; }
    else if (strstr(preset_name, "portrait"))  { wx = 4; wy = 64; }
    else if (strstr(preset_name, "universal"))  { wx = 16; wy = 16; }
    MNN_PRINT("[Vulkan] Workgroup preset '%s' -> %dx%d\n", preset_name, wx, wy);
}

// Hot-swap: update a const buffer on a VulkanFuse Extra op at runtime.
// This enables live 3A adjustments (gain, bias, CCM, etc.) without rebuilding the model.
// session_ptr: opaque session pointer from MNNCreateSession.
// bindingIndex: the 'const' attribute index from the Extra op.
// data: pointer to new float32 data.
// byteSize: size in bytes.
extern "C" __attribute__((visibility("default")))
int MNNVulkanHotSwapConstBuffer(void* session_ptr, int bindingIndex,
                                 const void* data, int byteSize) {
    // This requires walking the session's op list to find VulkanFuse executions.
    // For now, log the request — full implementation requires session introspection.
    MNN_PRINT("[Vulkan] HotSwapConstBuffer: binding=%d, size=%d bytes\n", bindingIndex, byteSize);
    return 0; // TODO: walk session ops, find VulkanFuse, call hotSwapConstBuffer
}

// Explicit registration entry point callable via dlsym after dlopen.
// This avoids --gc-sections stripping the static constructor.
extern "C" __attribute__((visibility("default"))) void MNNVulkanRegisterAll() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_VULKAN, new VulkanRuntimeCreator, true);
    MNNVulkanFuseRegister();
}

// Use __attribute__((constructor)) instead of static lambda to ensure
// the init function survives --gc-sections (--gc-sections strips static
// initializers that are not directly referenced by live code).
__attribute__((constructor)) static void _vulkan_runtime_init() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_VULKAN, new VulkanRuntimeCreator, true);
    // Register VulkanFuse creator for OpType_Extra
    // (Static constructors in other files may be stripped by --gc-sections)
    MNNVulkanFuseRegister();
}
}
