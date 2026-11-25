//
//  VulkanFuse.cpp
//  MNN
//
//  Created by MNN on 2023/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "component/MNNIspTrace.hpp"
#include <android/log.h>
#include <atomic>
#include <sys/system_properties.h>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
// Explicitly include buffer backend's VulkanBackend.hpp (has getPipelineFactory/getBuffer)
// to avoid image/backend shadowing it.
#include "../backend/VulkanBackend.hpp"
#include "VulkanBasicExecution.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanFuse.hpp"
#include "IspSpvLookup.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
#include <stdio.h>
#include <chrono>
#include <cstdio>
#include <stdlib.h>
namespace MNN {

// Perf-build gate for the per-dispatch VulkanFuse diagnostics. In a full-speed
// perf run these fprintf(stderr) calls (~14 ops x ~10 lines per frame on a 30fps
// camera stream) dominate host overhead; they are opt-in via ISP_DEBUG_VLOG so
// the default build is silent + fast. Default OFF.
// Trace sink: opened lazily when the gate is on; Termux reads the file.
static FILE* s_traceFile = nullptr;
static void ensureTraceFile() {
    if (!s_traceFile) {
        s_traceFile = fopen("/sdcard/Download/isp_vlog_trace", "a");
    }
}

// [PROBE] shared frame counter for the one-shot selective probe; reset when
// the trace re-arms (each pipeline rebuild) so post-rebuild frames get probed.
static std::atomic<int> sProbeFrameCount{0};

// Trace sink accessor for other TUs (VulkanBasicExecution per-frame trace).
extern "C" __attribute__((visibility("default"))) FILE* mnnIspTraceSink() { return s_traceFile; }
int ispProbeFrameTake() { return sProbeFrameCount.fetch_add(1); }
bool ispProbeFramesLeft(int limit) { return sProbeFrameCount.load() < limit; }

// Runtime override (JNI): setprop/file unnecessary — broadcast-toggleable.
static std::atomic<bool> s_vlogOverride{false};
void ispResetProbeCounter() { sProbeFrameCount.store(0); }

extern "C" __attribute__((visibility("default"))) void ispSetVlog(bool on) {
    s_vlogOverride.store(on);
    if (on) ensureTraceFile();
    if (on) { sProbeFrameCount.store(0); }
    if (s_traceFile) { fprintf(s_traceFile, "[VulkanFuse] ispSetVlog(%d) — trace armed\n", on ? 1 : 0); fflush(s_traceFile); }
    // Also mirror into the property so Termux getprop sees the state.
    __system_property_set("debug.isp_vlog", on ? "1" : "0");
    __android_log_print(ANDROID_LOG_INFO, "VulkanFuse", "ispSetVlog: %d", on ? 1 : 0);
}

static inline bool ispVlog() {
    // Env OR system property (setprop debug.isp_vlog 1) — the property is
    // readable at runtime from Termux without an app restart.
    static const bool envV = [] {
        const char* e = getenv("ISP_DEBUG_VLOG");
        return e && strcmp(e, "0") != 0;
    }();
    if (envV) return true;
    if (s_vlogOverride) return true;
    // System property debug.isp_vlog — the APP sets it via
    // SystemProperties.set() reflection (works for debuggable apps); read
    // cheaply every call.
    char propVal[PROP_VALUE_MAX] = {0};
    __system_property_get("debug.isp_vlog", propVal);
    bool on = propVal[0] != '\0' && strcmp(propVal, "0") != 0;
    if (on && !s_traceFile) ensureTraceFile();
    return on;
}


#define ISP_VLOG(...) do { if (ispVlog()) { \
        char _vbuf[512]; snprintf(_vbuf, sizeof(_vbuf), __VA_ARGS__); \
        fprintf(stderr, "%s", _vbuf); fflush(stderr); \
        __android_log_print(ANDROID_LOG_INFO, "VulkanFuse", "%s", _vbuf); \
        if (s_traceFile) { fprintf(s_traceFile, "%s", _vbuf); fflush(s_traceFile); } } } while (0)

// Helper to log stage elapsed time with stage label
static void _logStage(const char* stage, const std::chrono::steady_clock::time_point& start) {
    if (!ispVlog()) return;
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    fprintf(stderr, "[VulkanFuse-STAGE] %s: %lld ms\n", stage, ms);
    fflush(stderr);
}

// Helper to log elapsed time since a given time point
static void _logElapsed(const char* label, const std::chrono::steady_clock::time_point& start) {
    if (!ispVlog()) return;
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    fprintf(stderr, "[VulkanFuse] %s: %lld ms\n", label, ms);
    fflush(stderr);
}

// ── SPIR-V binding discovery ─────────────────────────────────────────────
//
// Reading the descriptor bindings out of the bytecode is the only way to keep
// the client-side map honest. The shader sources and the compiled `.spv` are
// separate files and drift; and of the 80 shipped `isp_*.spv`, 67 declare
// `InBuf@3`/`OutBuf@4` while 12 still declare the older `InBuf@1`/`OutBuf@2`.
// A single hardcoded map is wrong for one group or the other, and getting it
// wrong is silent: the op reads a zero-filled buffer and writes where nobody
// reads, so the frame keeps whatever the previous stage left.
//
// The scan walks the instruction stream for `OpName` / `OpDecorate Binding` /
// `OpTypePointer` / `OpVariable` — the same four records `spirv-dis` prints.
namespace {

struct IspSpvBindings {
    int in = -1;      // binding of the variable whose struct is named InBuf/FrameIn
    int out = -1;     // ...OutBuf/FrameOut
    int maxBinding = -1;
    /// Local workgroup size from `OpExecutionMode ... LocalSize`, 0 when the
    /// module does not declare one (specialisation-constant shaders).
    int localX = 0;
    int localY = 0;
    bool valid = false;
};

/// True when the SPIR-V word stream is well formed enough to scan.
bool spvUsable(const uint8_t* data, size_t bytes) {
    return data != nullptr
        && bytes >= 20
        && (bytes % 4) == 0
        && reinterpret_cast<const uint32_t*>(data)[0] == 0x07230203u;
}

IspSpvBindings scanSpvBindings(const uint8_t* data, size_t bytes) {
    IspSpvBindings r;
    if (!spvUsable(data, bytes)) {
        return r;
    }
    const uint32_t* w = reinterpret_cast<const uint32_t*>(data);
    const size_t n = bytes / 4;

    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, int> bindings;
    std::unordered_map<uint32_t, uint32_t> pointee;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> vars; // type, id, storage

    size_t i = 5; // skip the 5-word header
    while (i < n) {
        const uint32_t wc = w[i] >> 16;
        const uint32_t op = w[i] & 0xFFFF;
        if (wc == 0 || i + wc > n) {
            break;
        }
        switch (op) {
            case 5: { // OpName target "literal"
                if (wc >= 3) {
                    char buf[64] = {0};
                    const size_t chars = (wc - 2) * 4;
                    memcpy(buf, &w[i + 2], chars < sizeof(buf) - 1 ? chars : sizeof(buf) - 1);
                    names[w[i + 1]] = std::string(buf, strnlen(buf, sizeof(buf)));
                }
                break;
            }
            case 71: // OpDecorate target decoration ...
                if (wc >= 4 && w[i + 2] == 33) { // Binding
                    bindings[w[i + 1]] = (int)w[i + 3];
                }
                break;
            case 32: // OpTypePointer result storage pointee
                if (wc >= 4) {
                    pointee[w[i + 1]] = w[i + 3];
                }
                break;
            case 59: // OpVariable ptrType result storageClass
                if (wc >= 4) {
                    vars.emplace_back(w[i + 1], w[i + 2], w[i + 3]);
                }
                break;
            case 16: // OpExecutionMode entryPoint mode [literals]
                // 17 == LocalSize, followed by x, y, z. The shader's declared
                // geometry is authoritative: dispatching with a different
                // workgroup size than the module was built for is undefined.
                if (wc >= 6 && w[i + 2] == 17) {
                    r.localX = (int)w[i + 3];
                    r.localY = (int)w[i + 4];
                }
                break;
            default:
                break;
        }
        i += wc;
    }

    for (auto& t : vars) {
        const uint32_t storage = std::get<2>(t);
        // 12 = StorageBuffer, 2 = Uniform. isp.pipeline declares its buffers as
        // plain Uniform, so both have to be considered.
        if (storage != 12 && storage != 2) {
            continue;
        }
        auto pt = pointee.find(std::get<0>(t));
        const uint32_t structId = (pt != pointee.end()) ? pt->second : 0;
        std::string name;
        auto ni = names.find(structId);
        if (ni != names.end()) {
            name = ni->second;
        } else {
            auto vi = names.find(std::get<1>(t));
            if (vi != names.end()) {
                name = vi->second;
            }
        }
        auto bi = bindings.find(std::get<1>(t));
        if (bi == bindings.end()) {
            continue;
        }
        const int b = bi->second;
        if (b > r.maxBinding) {
            r.maxBinding = b;
        }
        if (name == "InBuf" || name == "FrameIn") {
            r.in = b;
        } else if (name == "OutBuf" || name == "FrameOut") {
            r.out = b;
        }
    }
    r.valid = (r.in >= 0 || r.out >= 0);
    return r;
}

} // namespace

VulkanFuse::VulkanFuse(const Extra* extra, Backend* bn, int inputSize, int outputSize) : VulkanBasicExecution(bn) {
    if (s_traceFile && extra->type()) {
        fprintf(s_traceFile, "[VulkanFuse] CREATED op type=%s\n", extra->type()->str().c_str());
        fflush(s_traceFile);
    }
    auto vkBn = static_cast<VulkanBackend*>(bn);
    auto factory = vkBn->getPipelineFactory();
    if (extra->type()) mType = extra->type()->str();
#ifdef ENABLE_VULKAN_TIME_PROFILE
    // Name the per-op GPU timing record after the real isp.* type ("isp.normalize"
    // etc.) so each stage is visible individually in the profile string instead
    // of being collapsed into a single "Extra" bucket. VulkanBackend::onCreate
    // preserves this name for OpType_Extra ops (does not overwrite it).
    if (extra && extra->type()) {
        setName(extra->type()->c_str());
    }
#endif
    mOutputBinding.resize(outputSize);
    mInputBinding.resize(inputSize);
    mGroupSize.resize(3);
    mGlobalSize.resize(3);
    // Detect single-workgroup reduction extras (reduce_keepdims attr)
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "reduce_keepdims" && (attr->b() || attr->i() != 0)) {
            mReduce = true;
            break;
        }
    }
    // Detect elementwise extras (output shape copies input shape).
    //
    // An explicit attribute wins, but the graph should not have to say it: a
    // shader that dispatches one thread per output pixel IS elementwise, and its
    // `OpExecutionMode LocalSize` says so. Requiring the attribute meant every
    // fused op had to be told twice, and a block that forgot it left W/H at zero
    // — the shader then early-returned on `x >= w` and wrote nothing.
    //
    // The scan below has not run yet at this point, so peek at it here: the
    // SPIR-V bytes are resolved later in the constructor, so this uses the
    // free-standing lookup directly.
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "elementwise" && (attr->b() || attr->i() != 0)) {
            mElementwise = true;
            break;
        }
    }
    if (!mElementwise && extra->type() && extra->type()->str().rfind("isp.", 0) == 0) {
        auto spvPeek = lookupIspSpv(extra->type()->str());
        if (spvPeek.data != nullptr) {
            auto peek = scanSpvBindings(spvPeek.data, (size_t)spvPeek.size);
            if (peek.localX > 1 && peek.localY > 0) {
                mElementwise = true;
                if (s_traceFile) {
                    fprintf(s_traceFile,
                            "[AUTODETECT] %s: elementwise from LocalSize %dx%d\n",
                            extra->type()->str().c_str(), peek.localX, peek.localY);
                    fflush(s_traceFile);
                }
            }
        }
    }
    // Detect planar-NCHW spatial ops (debayer/display): their INPUT data is
    // channel-plane NCHW even though session tensors are tagged NHWC, so the
    // W/H derivation must ignore the format tag and use dims[-1]/dims[-2].
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "spatial_nchw" && (attr->b() || attr->i() != 0)) {
            mSpatialNchw = true;
            break;
        }
    }
    // Find shader
    const uint8_t* data = nullptr;
    size_t dataSize = 0;
    // [ISP EMBEDDED SPIRV] Check embedded lookup first
    if (extra->type() && extra->type()->str().rfind("isp.", 0) == 0) {
        auto spvData = lookupIspSpv(extra->type()->str());
        if (spvData.data != nullptr && spvData.size > 0) {
            data = spvData.data;
            dataSize = spvData.size;
        }
    }
    // If not found in embedded lookup, check Extra op attributes
    if (data == nullptr) {
        for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "spirv") {
            // Try tensor format first (int8s)
            if (attr->tensor() && attr->tensor()->int8s() && attr->tensor()->int8s()->size() > 0) {
                data = (uint8_t*)attr->tensor()->int8s()->data();
                dataSize = attr->tensor()->int8s()->size();
                break;
            }
            // Fallback: check string list (ONNX STRING attribute converted to string list)
            if (attr->list() && attr->list()->s() && attr->list()->s()->size() > 0) {
                const ::flatbuffers::String* fbStr = attr->list()->s()->GetAsString(0);
                if (fbStr) {
                    data = (const uint8_t*)fbStr->c_str();
                    dataSize = fbStr->size();
                }
                break;
            }
            // Fallback: check single string attribute
            if (attr->s() && attr->s()->size() > 0) {
                auto str = attr->s()->str();
                data = (const uint8_t*)str.c_str();
                dataSize = str.size();
                break;
            }
        }
    }
    }
    // Caller (VulkanFuseCreator::onCreate) is responsible for validating
    // that SPIR-V is present before constructing VulkanFuse. Reaching here
    // with data==nullptr is a programmer error; fall through to the rest of
    // the constructor which will fail loudly at createComputePipeline().

    // Helper lambda to read ints from either list or tensor attribute
    auto readInts = [](const Attribute* attr, int* out, int count) -> bool {
        if (attr->list() && attr->list()->i()) {
            auto data = attr->list()->i();
            for (int j = 0; j < count && j < data->size(); j++) {
                out[j] = data->data()[j];
            }
            return true;
        }
        if (attr->tensor()) {
            if (attr->tensor()->int32s()) {
                auto data = attr->tensor()->int32s();
                for (int j = 0; j < count && j < data->size(); j++) {
                    out[j] = data->data()[j];
                }
                return true;
            }
            if (attr->tensor()->int8s()) {
                auto data = attr->tensor()->int8s();
                for (int j = 0; j < count && j < data->size(); j++) {
                    out[j] = data->data()[j];
                }
                return true;
            }
        }
        return false;
    };
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "global_size") {
            readInts(attr, mGlobalSize.data(), 3);
            mNeedAutoTuning = true;
            break;
        }
    }
    // Read preferred workgroup size from group_size attribute.
    // This is the LOCAL_SIZE the shader uses (e.g. 16×16).
    // optimized_dispatch uses this to compute dispatch group count.
    mPreferredLocalSize = {16, 16, 1};
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "group_size") {
            readInts(attr, mPreferredLocalSize.data(), 3);
            mNeedAutoTuning = false;
            break;
        }
    }
    // Optimization: Check for minimal dispatch mode
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "optimized_dispatch") {
            mOptimizedDispatch = attr->b();
            break;
        }
    }
    // Early-Z: Skip workgroups entirely outside valid image bounds.
    // Attr "early_z" = true, "valid_bounds" = {x0, y0, x1, y1} in pixels.
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "early_z") {
            mEarlyZ = attr->b();
        }
        if (attr->key()->str() == "valid_bounds") {
            if (attr->list() && attr->list()->i()) {
                auto data = attr->list()->i();
                for (int j = 0; j < 4 && j < data->size(); j++) {
                    mValidBounds.push_back(data->data()[j]);
                }
            }
        }
    }
    // FP16 const packing: when fp16_consts=true, pack const buffers as FP16
    // to halve GPU→shader bandwidth. Shader reads via unpackHalf2x16.
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "fp16_consts") {
            mFp16Consts = attr->b();
            break;
        }
    }
    // [VulkanFuse] mOptimizedDispatch set
    // ... rest of constructor ...
    std::vector<VkDescriptorType> types;
    int maxIndex = -1;
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "input") {
            auto list = attr->list()->i()->data();
            int binding = (int)list[1];
            maxIndex = ALIMAX(maxIndex, binding);
        } else if (attr->key()->str() == "const") {
            maxIndex = ALIMAX(maxIndex, (int)attr->i());
        }
    }
    // [FIX] VkDescriptorType's zero value is VK_DESCRIPTOR_TYPE_SAMPLER, so any
    // binding at or below maxIndex that no `input`/`const` attr writes keeps
    // SAMPLER while its SPIR-V declares a storage buffer — the descriptor layout
    // and the shader then disagree and vkCreateComputePipelines/the driver
    // rejects the op. isp.pipeline is exactly that case: it declares binding 0
    // (const) and 3..N (its coefficient inputs), and the two reserved slots it
    // uses — binding 1 DEBUG TAP and binding 2 CONFIG — are written by NEITHER,
    // so they stayed SAMPLER. Default every slot to STORAGE_BUFFER first; the
    // attribute loops below overwrite the ones that declare otherwise.
    types.assign(maxIndex + 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    std::vector<std::tuple<int, void*, size_t>> constStoragePtrs;
    std::vector<std::tuple<int, void*, size_t>> constUniformPtrs;
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "input") {
            auto list = attr->list()->i()->data();
            if (list[1] >= 0) {
                if (0 == list[0]) {
                    mInputBinding[attr->i()] = list[1];
                } else {
                    mOutputBinding[attr->i()] = list[1];
                }
            }
            {
                int binding = (int)list[1];
                if (binding >= 0 && binding < types.size()) {
                    if (attr->b()) {
                        types[binding] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    } else {
                        types[binding] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    }
                }
            }
            continue;
        }
        if (attr->key()->str() == "const") {
            auto b = attr->tensor();
            void* result = nullptr;
            size_t bufferSize = 0;
            switch (b->dataType()) {
                case DataType_DT_FLOAT:
                    result = (void*)b->float32s()->Data();
                    bufferSize = b->float32s()->size() * sizeof(float);
                    break;
                case DataType_DT_INT32:
                    result = (void*)b->int32s()->Data();
                    bufferSize = b->int32s()->size() * sizeof(float);
                    break;
                default:
                    MNN_ASSERT(false);
                    break;
            }
            // FP16 const packing: when fp16_consts=true, convert f32→f16
            // to halve const buffer bandwidth. Shader reads via unpackHalf2x16.
            if (mFp16Consts && b->dataType() == DataType_DT_FLOAT) {
                int count = b->float32s()->size();
                // Pack as FP16: each float → half (2 bytes)
                // Store as array of uint32 where each holds 2 packed halfs
                std::vector<uint16_t> fp16(count);
                for (int j = 0; j < count; j++) {
                    float v = b->float32s()->data()[j];
                    // Simple round-to-nearest FP16
                    uint32_t f = *((uint32_t*)&v);
                    uint32_t sign = (f >> 16) & 0x8000;
                    int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
                    uint32_t mantissa = (f >> 13) & 0x3FF;
                    if (exponent <= 0) { fp16[j] = (uint16_t)sign; }
                    else if (exponent >= 31) { fp16[j] = (uint16_t)(sign | 0x7BFF); }
                    else { fp16[j] = (uint16_t)(sign | (exponent << 10) | mantissa); }
                }
                result = fp16.data();
                bufferSize = count * sizeof(uint16_t);
                // Store fp16 data to keep alive during merge
                mFp16DataStorage = std::vector<uint8_t>((uint8_t*)fp16.data(), (uint8_t*)fp16.data() + bufferSize);
                result = mFp16DataStorage.data();
            }
            if (attr->b()) {
                types[attr->i()] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                constUniformPtrs.emplace_back(std::make_tuple(attr->i(), result, bufferSize));
            } else {
                types[attr->i()] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                constStoragePtrs.emplace_back(std::make_tuple(attr->i(), result, bufferSize));
            }
            continue;
        }
    }
    auto alignSize = vkBn->device().proty().limits.minMemoryMapAlignment;
    size_t offset = 0;
    std::shared_ptr<VulkanCommandPool::Buffer> cmdbuffer( vkBn->getPool().allocBuffer());
    cmdbuffer->begin(0);
    // DEBUG TAP (binding 1) and CONFIG (binding 2) sizes, needed both inside the
    // merge lambda (to reserve the bytes) and after it (to register the slots).
    // Hoisted here because they are per-instance constants, not per-merge.
    const size_t kTapBytes = (64u + 64u * 64u) * sizeof(float);   // header + 64x64 payload
    const size_t kCfgBytes = 256u;
    auto merge = [&](const std::vector<std::tuple<int, void*, size_t>>& constPtrs, VkDescriptorType type) {
        if (constPtrs.empty()) {
            return std::make_tuple(std::vector<std::tuple<int, size_t, size_t>>{}, std::shared_ptr<VulkanBuffer>(nullptr), std::shared_ptr<VulkanBuffer>(nullptr));
        }
        std::vector<std::tuple<int, size_t, size_t>> mConstOffset;
        for (auto& constAttr : constPtrs) {
            // Optimization: Power-of-2 alignment for faster GPU access
            auto size = UP_DIV(std::get<2>(constAttr), alignSize) * alignSize;
            size = std::max<size_t>(size, 256); // Minimum 256-byte alignment
            mConstOffset.emplace_back(std::make_tuple(std::get<0>(constAttr), size, offset));
            offset += size;
        }
        // Reserve the DEBUG TAP (binding 1) and CONFIG (binding 2) slots for
        // isp.pipeline BEFORE the host buffer is sized: the tap is 64 header +
        // 64*64 payload floats (16 KB), the config 256 B. Both are zeroed by the
        // existing memset, so an unwritten tap reads as "op has not run" and an
        // unwritten config reads as "normal behaviour" (cfg[0] >= 0.5 bypasses).
        if (!mType.empty() && mType == "isp.pipeline") {
            offset += kTapBytes + kCfgBytes;
        }
        std::shared_ptr<VulkanBuffer> hostBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, offset, nullptr, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
        auto ptr = (uint8_t*)hostBuffer->map();
        for (int i=0; i<constPtrs.size(); ++i) {
            ::memcpy(ptr + std::get<2>(mConstOffset[i]), std::get<1>(constPtrs[i]), std::get<2>(constPtrs[i]));
        }
        hostBuffer->unmap();
        VkBufferUsageFlags bufferUsage = (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) 
            ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT 
            : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        std::shared_ptr<VulkanBuffer> vkBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, offset, nullptr, 
            bufferUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
        VkBufferCopy bufferCopy;
        bufferCopy.size = offset;
        bufferCopy.dstOffset = 0;
        bufferCopy.srcOffset = 0;
        vkCmdCopyBuffer(cmdbuffer->get(), hostBuffer->buffer(), vkBuffer->buffer(),
                        1, &bufferCopy);
        return std::make_tuple(mConstOffset, vkBuffer, hostBuffer);
    };
    mConstStorageOffset.clear();
    mConstUniformOffset.clear();
    auto uniforms = merge(constUniformPtrs, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    mConstUniformOffset = std::get<0>(uniforms);
    mConstUniformBuffer = std::get<1>(uniforms);
    mConstUniformHostBuffer = std::get<2>(uniforms);
    auto storages = merge(constStoragePtrs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    mConstStorageOffset = std::get<0>(storages);
    mConstStorageBuffer = std::get<1>(storages);
    mConstStorageHostBuffer = std::get<2>(storages);
    cmdbuffer->end();
    auto fence = vkBn->getPool().submit(cmdbuffer->get());

    // [ISP] Direct-emit fallback. The converter's IspOpConverter (run at
    // optimizeLevel=0) copies the ONNX isp.* op's attributes verbatim, but the
    // direct-emit path (OnnxGraphComposer) emits the isp.* ops WITHOUT the
    // input/const/global_size binding attributes that VulkanFuse normally reads.
    // lookupIspSpv() still resolves the (SPIR-V-baked) shader by type, but
    // `types`/`mInputBinding`/`mOutputBinding` come back EMPTY -> the
    // descriptor-set layout built from `types` has ZERO bindings, and the first
    // DescriptorSet::writeBuffer(bind) reads mBufferTypes[bind] out-of-bounds,
    // yielding a garbage descriptor type (0xFFFFFFFF). Mesa/freedreno then
    // dereferences a null descriptor-set-layout entry inside
    // vkUpdateDescriptorSets -> SIGSEGV, fault_addr=0x18.
    //
    // All 50 embedded isp.* shaders share an identical, verified convention
    // (dumped from every isp_*_spv.h via spirv-dis): LocalSize 16x16x1;
    // binding 0 = const/param SSBO (read-only), binding 1 = input frame
    // (NonWritable StorageBuffer), binding 2 = output frame (NonReadable
    // StorageBuffer); all BufferBlock/STORAGE_BUFFER. Populate the layout and
    // the tensor->binding map from this convention so the ctor-built
    // descriptor set is valid. Extra scalar inputs (e.g. isp.normalize's
    // sensor_max) have no SPIR-V storage binding -> bound to a reserved
    // binding 3 the shader never references, so the descriptor is valid and the
    // shader never dereferences them.
    int spvMaxBinding = -1; // highest binding the shader declares, from the scan
    // Scan unconditionally: the map below is only built when the graph left the
    // bindings empty, but the layout ceiling and the frame-slot override are
    // needed either way. `data`/`dataSize` are resolved above regardless.
    IspSpvBindings spvBindEarly = scanSpvBindings(data, dataSize);
    spvMaxBinding = spvBindEarly.maxBinding;
    if (types.empty() && !mType.empty() && mType.rfind("isp.", 0) == 0) {
        // UNIVERSAL isp.* binding map — one positional rule for ALL shaders
        // (verified against every isp_*.comp SPIR-V dump):
        //   binding 0        = params SSBO (the shader's scalar Uniforms block:
        //                      width/height + per-op coefficients. NEVER a frame,
        //                      never metadata — pure scalars.)
        //   binding 1        = input[0]  — ALWAYS the input frame (RO)
        //   binding 2        = output[0] — ALWAYS the output frame (WO)
        //   binding 3..      = EVERYTHING ELSE, extra inputs first, then extra
        //     outputs. The two streams share one numbering space and never
        //     overlap:
        //        input[n]  (n>=1) -> 3 + (n-1)
        //        output[m] (m>=1) -> 3 + (inputSize - 1) + (m-1)
        //     A metadata input (CCM matrix, BLC gains, LSC grid, sensor_max…)
        //     therefore takes 3, and a metadata output only starts after all
        //     metadata inputs — so an op with BOTH never aliases the same slot.
        // isp.crop_blc_norm_cfa was retargeted to this map (out 3->2, BLC
        // param 2->3) so the entire opset shares a single convention.
        const int extraInputs  = inputSize  > 0 ? inputSize  - 1 : 0;
        const int extraOutputs = outputSize > 0 ? outputSize - 1 : 0;
        // UNIVERSAL isp.* BINDING MAP — one rule for every shader.
        //
        //   0        params SSBO  (scalar uniforms; W/H hot-swapped into 0/1)
        //   1        DEBUG TAP    (isp.pipeline only: the op's output snapshot)
        //   2        CONFIG       (isp.pipeline only: cfg[0] >= 0.5 bypasses)
        //   3        input[0]     — the frame
        //   4        output[0]    — the frame
        //   5..      extras: metadata inputs first, then metadata outputs
        //
        // Verified against the shipped bytecode, not against the sources: of the
        // 80 real `isp_*.spv`, 66 declare `InBuf@3` and `OutBuf@4`. Exactly one
        // (`isp_crop_debug`) disagrees, and 13 have no frame tensor at all (the
        // `algo_*` statistics ops bind named buffers and never enter this path).
        //
        // This was previously split into a "gold" map for `isp.pipeline` and a
        // frame-at-1/2 map for everything else, on the belief that the other
        // shaders declared `InBuf@1`/`OutBuf@2`. They do not. The consequence
        // was that every fused op read the zero-filled DEBUG TAP as its input
        // and wrote its result into CONFIG, which nothing reads — the graph
        // output kept whatever the previous stage had left there.
        //
        // The layout size was wrong in the same way: with `kFirstExtra` at 3 a
        // 1-in/1-out op produced `types.size() == 3`, covering bindings 0..2, so
        // the descriptor writes for 3 and 4 landed past the end of
        // `mBufferTypes`.
        //
        // `isp.pipeline` keeps the reserved tap/config slots, but those are
        // appended as extra const-storage entries (the `mConstStorageOffset`
        // block below); they do not move the frame.
        // Ask the bytecode where it put them, and only fall back to the
        // universal map when the scan finds nothing (or the module is malformed).
        // 67 of the 80 shipped shaders answer 3/4; twelve still answer 1/2, and
        // those twelve silently read a zero-filled buffer under a fixed map.
        IspSpvBindings& spvBind = spvBindEarly;
        int kFrameInBinding  = spvBind.in  >= 0 ? spvBind.in  : 3;
        int kFrameOutBinding = spvBind.out >= 0 ? spvBind.out : 4;
        // Metadata starts after whichever slot is higher, so the frame can never
        // be overwritten by an extra input.
        auto kFirstExtra = (uint32_t)((kFrameInBinding > kFrameOutBinding
                                       ? kFrameInBinding : kFrameOutBinding) + 1);
        if (s_traceFile) {
            fprintf(s_traceFile,
                    "[BINDINGS] %s: shader says in@%d out@%d (scanned=%d, highest=%d), "
                    "extras start at %u\n",
                    mType.c_str(), kFrameInBinding, kFrameOutBinding,
                    (int)spvBind.valid, spvBind.maxBinding, kFirstExtra);
            fflush(s_traceFile);
        }
        const int maxBind = kFirstExtra - 1 + extraInputs + extraOutputs;
        types.assign(maxBind + 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        for (int i = 0; i < inputSize; ++i) {
            // frame -> kFrameInBinding ; extra metadata inputs -> kFirstExtra, ...
            mInputBinding[i] = (i == 0) ? kFrameInBinding : kFirstExtra + (i - 1);
        }
        for (int i = 0; i < outputSize; ++i) {
            // frame -> kFrameOutBinding ; extra metadata outputs -> kFirstExtra + extraInputs, ...
            mOutputBinding[i] = (i == 0) ? kFrameOutBinding : kFirstExtra + extraInputs + (i - 1);
        }
        // Binding 0 param SSBO, binding 1 DEBUG TAP, binding 2 CONFIG: all
        // zero-filled, host-visible storage buffers, so every shader reads valid
        // data (params zero -> early-return; tap starts empty; config zero ->
        // normal behaviour, since cfg[0] >= 0.5 is what bypasses).
        //
        // The tap is 64 header + 64*64 payload floats = 16 KB. Allocated ONCE
        // per op instance and reused every frame: the shader overwrites it, and
        // a reader only needs the latest frame.
        const size_t cbsz = 256;
        mConstStorageBuffer = std::make_shared<VulkanBuffer>(
            vkBn->getMemoryPool(), false, cbsz, nullptr,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        mConstStorageHostBuffer = mConstStorageBuffer;
        auto* cbp = (uint8_t*)mConstStorageBuffer->map();
        if (cbp) { ::memset(cbp, 0, cbsz); mConstStorageBuffer->unmap(); }
        mConstStorageOffset.clear();
        mConstStorageOffset.emplace_back(std::make_tuple(0, cbsz, size_t(0)));
        if (mType == "isp.pipeline") {
            // Appended AFTER the params entry, so their offsets follow it. The
            // merge lambda reserved the bytes, so these fit.
            size_t tapOff = UP_DIV(cbsz, alignSize) * alignSize;
            mConstStorageOffset.emplace_back(std::make_tuple(1, kTapBytes, tapOff));
            mConstStorageOffset.emplace_back(std::make_tuple(2, kCfgBytes, tapOff + kTapBytes));
        }
        mPreferredLocalSize = {16, 16, 1};
        mOptimizedDispatch = true;
        mNeedAutoTuning = false;   // fixed LocalSize; no pipeline rebuild
    }
    // ── Frame slots follow the shader, not the graph ────────────────────
    //
    // The ONNX node may pin bindings with explicit `input` attributes, but those
    // are written by hand and go stale when a shader is rebuilt against a
    // different map. The bytecode is authoritative: whatever it declares for
    // `InBuf`/`OutBuf` is where the frame has to go.
    //
    // This also shifts the metadata inputs when the frame moved, so an extra
    // input can never land on the frame slot. Metadata is numbered from
    // `kFirstExtra`, which is recomputed from the highest frame slot.
    if (spvBindEarly.in >= 0 || spvBindEarly.out >= 0) {
        const int inSlot  = spvBindEarly.in  >= 0 ? spvBindEarly.in  : 3;
        const int outSlot = spvBindEarly.out >= 0 ? spvBindEarly.out : 4;
        if (inputSize > 0) {
            mInputBinding[0] = inSlot;
        }
        if (outputSize > 0) {
            mOutputBinding[0] = outSlot;
        }
        const int firstExtra = (inSlot > outSlot ? inSlot : outSlot) + 1;
        for (int i = 1; i < inputSize; ++i) {
            mInputBinding[i] = firstExtra + (i - 1);
        }
        for (int i = 1; i < outputSize; ++i) {
            mOutputBinding[i] = firstExtra + (inputSize > 0 ? inputSize - 1 : 0) + (i - 1);
        }
        if (s_traceFile) {
            fprintf(s_traceFile,
                    "[BINDINGS] %s: frame forced to in@%d out@%d, extras from %d\n",
                    mType.c_str(), inSlot, outSlot, firstExtra);
            fflush(s_traceFile);
        }
        // Grow the layout to cover everything that will be written.
        size_t want = (size_t)firstExtra + (size_t)(inputSize > 0 ? inputSize - 1 : 0)
                    + (size_t)(outputSize > 0 ? outputSize - 1 : 0);
        if (!mInputBinding.empty()) {
            want = std::max<size_t>(want, (size_t)mInputBinding.back() + 1);
        }
        if (!mOutputBinding.empty()) {
            want = std::max<size_t>(want, (size_t)mOutputBinding.back() + 1);
        }
        if (types.size() < want) {
            types.resize(want, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
    }

    // Layout adequacy guard.
    //
    // The descriptor layout must cover every binding the SPIR-V declares, and
    // the code that builds `types` cannot know that: the ONNX node's
    // `input`/`const` attributes describe only the tensors the graph wires up,
    // not the reserved slots a shader may hardcode. Two cases used to need
    // special handling — `isp.demosaic_g2_ccm` (a CCM slot the graph may omit)
    // and the identity-CCM identity bind — and both were patched by name.
    //
    // With the universal map the arithmetic covers frame+metadata for the
    // declared tensors, so the remaining risk is a shader that hardcodes a slot
    // beyond them. Grow to a fixed ceiling instead of guessing per op:
    // `isp_algo_gamma_scalar` is the widest shader in the set at binding 10.
    //
    // Writing past `types.size()` is not a soft failure: `writeBuffer` indexes
    // `mBufferTypes[bind]`, so an undersized layout is an out-of-bounds read
    // that yields a garbage descriptor type and a driver-side fault.
    //
    // The ceiling comes from the scan: the highest binding the module declares,
    // plus the reserved tap/config pair for `isp.pipeline`. Anything above
    // whatever the arithmetic produced is filled with storage buffers.
    if (!mType.empty() && mType.rfind("isp.", 0) == 0 && spvMaxBinding >= 0) {
        const size_t want = (size_t)spvMaxBinding + 1;
        if (types.size() < want) {
            if (s_traceFile) {
                fprintf(s_traceFile, "[LAYOUT] %s: growing descriptor layout %zu -> %zu\n",
                        mType.c_str(), types.size(), want);
                fflush(s_traceFile);
            }
            types.resize(want, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
    }
    mPipeline = factory->createComputePipeline(data, dataSize, types, std::vector<uint32_t>{});
    // createComputePipeline returns nullptr when the device rejects the shader
    // (VK_ERROR_INVALID_SHADER_NV / OOM / a driver without the needed feature).
    // Dereferencing it here faults at [null + 0x40] inside
    // VulkanPipeline::createSet (the SharedPtr<VulkanLayout> member), which is
    // exactly the device SIGSEGV: fault_addr=0x40, pc=createSet+8. Leave
    // mPipeline/mDescriptorSet null so the Creator can reject the op and let
    // MNN fall back to the CPU backend instead of crashing the process.
    if (!mPipeline.get()) {
        // MNN_ERROR so the message lands on the MNNJNI logcat tag (captured by
        // the app's MNN_INTERNAL tee), not just raw stderr which the app does
        // not persist. Knowing WHICH isp.* op failed pipeline creation is the
        // fastest way to pinpoint a bad shader on a new device.
        MNN_ERROR("[VulkanFuse] pipeline creation FAILED for %s — op will not run\n",
                  mType.c_str());
        fence->wait();
        return;
    }
    mDescriptorSet = mPipeline->createSet();
    fence->wait();
}

VulkanFuse::~VulkanFuse() {
    mDescriptorSet = nullptr;
}

ErrorCode VulkanFuse::hotSwapConstBuffer(int bindingIndex, const void* data, size_t byteSize) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto cmdbuffer = std::shared_ptr<VulkanCommandPool::Buffer>(vkBn->getPool().allocBuffer());
    cmdbuffer->begin(0);

    // When fp16_consts=true the const buffer holds PACKED FP16 values
    // (shader reads via unpackHalf2x16). Convert the input floats to FP16
    // so the hot-swapped bytes match the buffer layout.
    std::vector<uint8_t> fp16Packed;
    const void* srcData = data;
    size_t srcSize = byteSize;
    if (mFp16Consts && (byteSize % sizeof(float)) == 0) {
        int count = (int)(byteSize / sizeof(float));
        std::vector<uint16_t> fp16(count);
        const float* fdata = (const float*)data;
        for (int j = 0; j < count; j++) {
            float v = fdata[j];
            uint32_t f = *((uint32_t*)&v);
            uint32_t sign = (f >> 16) & 0x8000;
            int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
            uint32_t mantissa = (f >> 13) & 0x3FF;
            if (exponent <= 0) { fp16[j] = (uint16_t)sign; }
            else if (exponent >= 31) { fp16[j] = (uint16_t)(sign | 0x7BFF); }
            else { fp16[j] = (uint16_t)(sign | (exponent << 10) | mantissa); }
        }
        fp16Packed.resize(count * sizeof(uint16_t));
        ::memcpy(fp16Packed.data(), fp16.data(), count * sizeof(uint16_t));
        srcData = fp16Packed.data();
        srcSize = count * sizeof(uint16_t);
    }

    // Find the matching const buffer offset
    for (auto& iter : mConstUniformOffset) {
        if (std::get<0>(iter) == bindingIndex) {
            auto bufSize = std::min(srcSize, std::get<1>(iter));
            // Create host-visible staging buffer
            auto hostBuf = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, bufSize, nullptr,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            auto ptr = hostBuf->map();
            ::memcpy(ptr, srcData, bufSize);
            hostBuf->unmap();
            // Copy to GPU const uniform buffer
            VkBufferCopy copy{};
            copy.size = bufSize;
            copy.dstOffset = std::get<2>(iter);
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdbuffer->get(), hostBuf->buffer(),
                           mConstUniformBuffer->buffer(), 1, &copy);
            cmdbuffer->end();
            auto fence = vkBn->getPool().submit(cmdbuffer->get());
            fence->wait();
            return NO_ERROR;
        }
    }
    for (auto& iter : mConstStorageOffset) {
        if (std::get<0>(iter) == bindingIndex) {
            auto bufSize = std::min(srcSize, std::get<1>(iter));
            auto hostBuf = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, bufSize, nullptr,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            auto ptr = hostBuf->map();
            ::memcpy(ptr, srcData, bufSize);
            hostBuf->unmap();
            VkBufferCopy copy{};
            copy.size = bufSize;
            copy.dstOffset = std::get<2>(iter);
            copy.srcOffset = 0;
            vkCmdCopyBuffer(cmdbuffer->get(), hostBuf->buffer(),
                           mConstStorageBuffer->buffer(), 1, &copy);
            cmdbuffer->end();
            auto fence = vkBn->getPool().submit(cmdbuffer->get());
            fence->wait();
            return NO_ERROR;
        }
    }
    cmdbuffer->end();
    return NOT_SUPPORT;
}

ErrorCode VulkanFuse::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) {
    ISP_VLOG("[VulkanFuse] onEncode START");
    auto t_encode_start = std::chrono::steady_clock::now();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    // [STAGE 1] TENSOR PREPARATION: Validate inputs/outputs before SPIR-V presubmit
    auto t_stage = std::chrono::steady_clock::now();
    ISP_VLOG("[VulkanFuse-STAGE] onEncode: TENSOR_PREPARATION start");
    // Validate input tensors
    if (inputs.empty()) {
        ISP_VLOG("[VulkanFuse-STAGE] onEncode: TENSOR_PREPARATION error: no inputs");
        return NOT_SUPPORT;
    }
    // Validate output tensors
    if (outputs.empty()) {
        ISP_VLOG("[VulkanFuse-STAGE] onEncode: TENSOR_PREPARATION error: no outputs");
        return NOT_SUPPORT;
    }
    _logStage("onEncode: TENSOR_PREPARATION", t_stage);
    // [STAGE 2] SPIR-V PRESUBMIT: Verify shader pipeline before dispatch
    t_stage = std::chrono::steady_clock::now();
    ISP_VLOG("[VulkanFuse-STAGE] onEncode: SPIRV_PRESUBMIT start");
    if (!mPipeline.get()) {
        ISP_VLOG("[VulkanFuse-STAGE] onEncode: SPIRV_PRESUBMIT error: no pipeline");
        return NOT_SUPPORT;
    }
    // Validate descriptor sets
    if (!mDescriptorSet.get()) {
        ISP_VLOG("[VulkanFuse-STAGE] onEncode: SPIRV_PRESUBMIT error: no descriptor set");
        return NOT_SUPPORT;
    }
    _logStage("onEncode: SPIRV_PRESUBMIT", t_stage);
    // ── Dynamic-size derivation ──
    // If the baked global_size is invalid (0/negative — from dynamic ONNX input
    // at convert time), derive the dispatch size from the ACTUAL input tensor
    // dims at runtime, and patch the const buffer's first two floats (W, H).
    // dims==1 (scalar op): treat as 1×1. dims==2: [H,W]. dims>=4: [N,C,H,W].
    int tw = 0, th = 0, tc = 0;
    if (!inputs.empty() && inputs[0] && inputs[0]->buffer().dimensions >= 1) {
        int dims = inputs[0]->buffer().dimensions;
        if (dims == 1) {
            tw = 1; th = 1;
        } else if (dims == 2) {
            th = inputs[0]->buffer().dim[0].extent;
            tw = inputs[0]->buffer().dim[1].extent;
        } else {
            // Layout-aware W/H derivation: ISP chain tensors are plain NCHW
            // (NHWC is retired — the main input is [1,1,H,W] full-res).
            // For NCHW the spatial dims live at [dims-2]=H,[dims-1]=W and
            // the channel count at [1].
            bool isNhwc = false;
            if (mSpatialNchw) {
                // Op declared its input as planar NCHW data: trust dims only.
                isNhwc = false;
            } else if (dims >= 4) {
                auto df = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
                isNhwc = (df == MNN_DATA_FORMAT_NHWC);
                if (!isNhwc && inputs[0]->buffer().dim[3].extent <= 8 &&
                    inputs[0]->buffer().dim[1].extent > 8) {
                    // Heuristic fallback when the describe tag is missing:
                    // a trailing dim <= 8 with large dim[1] means channels-last.
                    isNhwc = true;
                }
            }
            if (isNhwc) {
                th = inputs[0]->buffer().dim[1].extent;        // H
                tw = inputs[0]->buffer().dim[2].extent;        // W
                tc = inputs[0]->buffer().dim[3].extent;        // C
            } else {
                tw = inputs[0]->buffer().dim[dims-1].extent;   // W
                th = inputs[0]->buffer().dim[dims-2].extent;   // H
                if (dims >= 4) tc = inputs[0]->buffer().dim[1].extent;  // C
            }
        }
    }
    if (mGlobalSize[0] <= 0 || mGlobalSize[1] <= 0) {
        if (tw > 0 && th > 0) {
            mGlobalSize[0] = tw;
            mGlobalSize[1] = th;
            mGlobalSize[2] = 1;
            mNeedAutoTuning = true;
        } else {
            // Never dispatch 0 workgroups — vkCmdDispatch requires ≥1. Fall
            // back to a single thread so the shader at least runs.
            mGlobalSize[0] = 1;
            mGlobalSize[1] = 1;
            mGlobalSize[2] = 1;
        }
    }
    // Elementwise extras: the real dispatch size and the shader's const W,H
    // must match the ACTUAL input tensor dims — otherwise the shader
    // early-returns on `x >= w || y >= h` and writes nothing.
    // Patch BOTH: override global_size from input dims AND hot-swap the
    // const buffer's first two floats (W,H).
    if (mElementwise && !mElementwisePatched && tw > 0 && th > 0) {
        ISP_VLOG("[VulkanFuse] ELEMENTWISE patch: tw=%d th=%d glob=[%d,%d,%d]\n",
            tw, th, mGlobalSize[0], mGlobalSize[1], mGlobalSize[2]);
        fflush(stderr);
        if (mGlobalSize[0] != tw || mGlobalSize[1] != th) {
            mGlobalSize[0] = tw;
            mGlobalSize[1] = th;
            mGlobalSize[2] = 1;
            mNeedAutoTuning = true;
        }
        float v[2] = { (float)tw, (float)th };
        ErrorCode hs = hotSwapConstBuffer(0, v, sizeof(v));
        ISP_VLOG("[VulkanFuse] ELEMENTWISE hotSwap done rc=%d\n", (int)hs);
        // Structural verification: dump the first 4 floats of the binding-0
        // const buffer AFTER the patch. Catches the W/H-offset class of bugs
        // (e.g. lsc_min had wh at offsets 4/5 — the patch wrote W/H into k1/k2
        // and left wh=(0,0) → shader early-returned → white screen).
        if (ispVlog() && !mConstStorageOffset.empty()) {
            auto& off0 = mConstStorageOffset[0];
            const float* cb = (const float*)mConstStorageBuffer->map();
            if (cb) {
                ISP_VLOG("[VulkanFuse] %s const[0..3] after ELEMENTWISE patch: [%.1f, %.1f, %.3f, %.3f] (expect [W,H,...])\n",
                         mType.c_str(), cb[0], cb[1], cb[2], cb[3]);
                mConstStorageBuffer->unmap();
            }
        }
        fflush(stderr);
        mElementwisePatched = true;
    }
    // Packed `isp.display_*` shaders read gamma from Uniforms[2] in the
    // binding-0 storage buffer, but the graph passes gamma as extra input[1]
    // (bound at binding 5 — the slot `isp.display` itself reads). Those packed
    // shaders never read binding 5, so without this copy u.gamma == 0 and
    // pow(x, 1e6) collapses every pixel to opaque black.
    if (mType.rfind("isp.display_", 0) == 0 && mConstStorageBuffer
        && inputs.size() > 1 && inputs[1]) {
        float gamma = 2.4f;
        auto* gp = inputs[1]->host<float>();
        if (gp && gp[0] > 0.01f) {
            gamma = gp[0];
        }
        auto* cb = reinterpret_cast<float*>(mConstStorageBuffer->map());
        if (cb) {
            cb[2] = gamma;
            mConstStorageBuffer->unmap();
        }
    }
    // Reduction extras: baked const {W,H(,C)} is 0 from dynamic input. Patch
    // ONCE (first encode) from the actual input tensor so the stride/reduce
    // loops run over the real image. ispc_stats const = {W,H,C};
    // calib_stats/af_focus = {W,H}. (Dims are stable for a fixed sensor.)
    if (mReduce && !mReducePatched && tw > 0 && th > 0) {
        if (mType == "isp.ispc_stats" && tc > 0) {
            float v[3] = { (float)tw, (float)th, (float)tc };
            hotSwapConstBuffer(0, v, sizeof(v));
        } else {
            float v[2] = { (float)tw, (float)th };
            hotSwapConstBuffer(0, v, sizeof(v));
        }
        mReducePatched = true;
    }
    // [STAGE 3] BIND DESCRIPTORS: Bind descriptors for SPIR-V execution
    // Per-frame tensor→descriptor binding: input/output tensors change
    // backing buffers every frame (pool churn), so write them into the
    // descriptor set on EVERY encode — a ctor-time bind would keep stale
    // device addresses alive after the first frame (black frames / rc=3).
    for (int i=0; i<(int)inputs.size(); ++i) {
        int binding = mInputBinding[i];
        // [ISP TRACE] Log deviceId for each input tensor — deviceId==0 means
        // CPU-allocated (dummy buffer) -> the GPU shader reads zeros.
        auto tensorBuffer = vkBn->getBuffer(inputs[i]);
        if (s_traceFile && inputs.size() <= 3) {
            fprintf(s_traceFile, "[DESCRIPTOR] %s input[%d] b%d devId=%p buf=%p backend=%p\n",
                mType.c_str(), i, binding, (void*)inputs[i]->deviceId(), (void*)std::get<0>(tensorBuffer), (const void*)vkBn);
            fflush(s_traceFile);
        }
        // [ISP-IN] staging alloc here; buffer captured at EXECUTE time.
        if (i == 0 && mType.find("isp.") == 0 && !mInProbeStage) {
            mInProbeStage = vkBn->createHostBuffer(64);
        }
        if (inputs[i]->deviceId() == 0) {
            if (s_traceFile) {
                fprintf(s_traceFile, "[DESCRIPTOR] WARNING: input[%d] deviceId=0 (CPU) -> dummy buffer for binding %d\n",
                    i, binding);
                fflush(s_traceFile);
            }
        }
        mDescriptorSet->writeBuffer(tensorBuffer, binding);
    }
    for (int i=0; i<(int)outputs.size(); ++i) {
        int binding = mOutputBinding[i];
        auto tensorBuffer = vkBn->getBuffer(outputs[i]);
        if (s_traceFile && outputs.size() <= 2 && mType.find("isp.") == 0) {
            fprintf(s_traceFile, "[DESCRIPTOR-OUT] %s output[%d] b%d devId=%p buf=%p\n",
                mType.c_str(), i, binding, (void*)outputs[i]->deviceId(), (void*)std::get<0>(tensorBuffer));
            fflush(s_traceFile);
        }
        mDescriptorSet->writeBuffer(tensorBuffer, binding);
        // [ISP-OUT] staging alloc here; buffer captured at EXECUTE time
        // (dynamic buffers churn per frame — an encode-time handle is stale).
        if (i == 0 && mType.find("isp.") == 0 && !mProbeStage) {
            mProbeStage = vkBn->createHostBuffer(192);
        }
    }
    if (mType == "isp.demosaic_g2_ccm" && inputs.size() < 2) {
        // Single-input form: no runtime CCM tensor. Bind a persistent identity
        // 3x3 matrix at the CCM slot so the shader reads a passthrough matrix
        // instead of an unbound descriptor (GPU fault / garbage colours).
        // `isp_demosaic_g2_ccm.spv` declares CcmBuf at binding 5, which is
        // `kFirstExtra` — the same slot a real CCM tensor would take. Created
        // once, reused every frame.
        if (!mIdentityCcmBuffer) {
            const size_t ccmBytes = 9 * sizeof(float);
            mIdentityCcmBuffer = std::make_shared<VulkanBuffer>(
                vkBn->getMemoryPool(), false, ccmBytes, nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_SHARING_MODE_EXCLUSIVE,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            float ident[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
            auto* p = (float*)mIdentityCcmBuffer->map();
            if (p) { ::memcpy(p, ident, ccmBytes); mIdentityCcmBuffer->unmap(); }
        }
        mDescriptorSet->writeBuffer(std::make_tuple(mIdentityCcmBuffer->buffer(),
            (VkDeviceSize)36, (VkDeviceSize)0), /* CCM slot */ 5);
    }
    for (auto& iter : mConstStorageOffset) {
        mDescriptorSet->writeBuffer(mConstStorageBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
    }
    for (auto& iter : mConstUniformOffset) {
        mDescriptorSet->writeBuffer(mConstUniformBuffer->buffer(), std::get<0>(iter), std::get<1>(iter), std::get<2>(iter));
    }
    if (mNeedAutoTuning && !mOptimizedDispatch) {
        // VulkanFuse dispatches a FUSED compute shader whose workgroup size is
        // FIXED (OpExecutionMode LocalSize) in the SPIR-V — 16x16x1 for all
        // isp.* ops, matching mPreferredLocalSize (group_size attr or the
        // {16,16,1} fallback). Compute the dispatch group count from the
        // shader-fixed local size; no pipeline rebuild needed.
        int lx = mPreferredLocalSize[0] > 0 ? mPreferredLocalSize[0] : 16;
        int ly = mPreferredLocalSize[1] > 0 ? mPreferredLocalSize[1] : 16;
        int lz = mPreferredLocalSize[2] > 0 ? mPreferredLocalSize[2] : 1;
        mGroupSize[0] = UP_DIV(mGlobalSize[0], lx);
        mGroupSize[1] = UP_DIV(mGlobalSize[1], ly);
        mGroupSize[2] = UP_DIV(mGlobalSize[2], lz);
        mNeedAutoTuning = false;
    } else if (mOptimizedDispatch) {
        // Group count = ceil(global_size / local_size).
        mGroupSize[0] = UP_DIV(mGlobalSize[0], ALIMAX(1, mPreferredLocalSize[0]));
        mGroupSize[1] = UP_DIV(mGlobalSize[1], ALIMAX(1, mPreferredLocalSize[1]));
        mGroupSize[2] = UP_DIV(mGlobalSize[2], ALIMAX(1, mPreferredLocalSize[2]));
        mNeedAutoTuning = false;
    }
    t_stage = std::chrono::steady_clock::now();
    ISP_VLOG("[VulkanFuse-STAGE] onEncode: BIND_DESCRIPTORS start");
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    _logStage("onEncode: BIND_DESCRIPTORS", t_stage);
    // [STAGE 4] DISPATCH: Issue GPU work to Vulkan queue
    t_stage = std::chrono::steady_clock::now();
    ISP_VLOG("[VulkanFuse-STAGE] onEncode: DISPATCH start");
    // Early-Z: Clamp dispatch to valid bounds, skipping entirely-outside workgroups.
    int dispatchX = mGroupSize[0], dispatchY = mGroupSize[1], dispatchZ = mGroupSize[2];
    if (mEarlyZ && mValidBounds.size() == 4 && mGlobalSize.size() >= 2) {
        int x0 = mValidBounds[0], y0 = mValidBounds[1];
        int x1 = mValidBounds[2], y1 = mValidBounds[3];
        int lsx = mGroupSize[0] > 0 ? UP_DIV(mGlobalSize[0], mGroupSize[0]) : 1;
        int lsy = mGroupSize[1] > 0 ? UP_DIV(mGlobalSize[1], mGroupSize[1]) : 1;
        // Compute dispatch origin and count for the valid region
        int gx0 = x0 / (mPreferredLocalSize[0] > 0 ? mPreferredLocalSize[0] : 16);
        int gy0 = y0 / (mPreferredLocalSize[1] > 0 ? mPreferredLocalSize[1] : 16);
        int gx1 = (x1 + mPreferredLocalSize[0] - 1) / (mPreferredLocalSize[0] > 0 ? mPreferredLocalSize[0] : 16);
        int gy1 = (y1 + mPreferredLocalSize[1] - 1) / (mPreferredLocalSize[1] > 0 ? mPreferredLocalSize[1] : 16);
        dispatchX = ALIMAX(1, gx1 - gx0);
        dispatchY = ALIMAX(1, gy1 - gy0);
    }
    auto t_disp0 = std::chrono::high_resolution_clock::now();
    vkCmdDispatch(cmdBuffer->get(), dispatchX, dispatchY, dispatchZ);
    // [ISP TRACE] Log the dispatch to the trace file for verification.
    if (s_traceFile) {
        fprintf(s_traceFile, "[DISPATCH] type=%s grid=[%d,%d,%d] wg=[%d,%d,%d] global=[%d,%d,%d]\n",
            mType.c_str(), dispatchX, dispatchY, dispatchZ,
            mPreferredLocalSize[0], mPreferredLocalSize[1], mPreferredLocalSize[2],
            mGlobalSize[0], mGlobalSize[1], mGlobalSize[2]);
        fflush(s_traceFile);
    }
    auto t_disp1 = std::chrono::high_resolution_clock::now();
    auto disp_us = std::chrono::duration_cast<std::chrono::microseconds>(t_disp1 - t_disp0).count();
    _logStage("onEncode: DISPATCH", t_stage);
    // Per-op GPU workload reporting (writes to stderr → logcat).
    //
    // Gated with everything else behind `MNN_ISP_TRACE`: a timing run wants the
    // frame loop silent, and 40 `fprintf(stderr)` calls during warmup are exactly
    // the kind of host work that shows up as a slow first frames.
    static thread_local int _log_counter = 0;
    const char* ispTrace = ::getenv("MNN_ISP_TRACE");
    const bool traceOn = ispTrace != nullptr && ispTrace[0] != '0';
    if (::getenv("ISP_DEBUG_VLOG") || (traceOn && _log_counter < 40)) {
        fprintf(stderr, "[VulkanFuse-PERF] type=%s W=%d H=%d groups=[%dx%dx%d] wg=%dx%dx%d submit_us=%lld pixels=%d\n",
            mType.c_str(), mGlobalSize[0], mGlobalSize[1],
            mGroupSize[0], mGroupSize[1], mGroupSize[2],
            dispatchX, dispatchY, dispatchZ,
            (int)disp_us, mGlobalSize[0] * mGlobalSize[1]);
        fflush(stderr);
        // Also emit via stderr (goes to logcat on Android)
        _log_counter++;
    }
    _logElapsed("[VulkanFuse] onEncode TOTAL", t_encode_start);
    return NO_ERROR;
}

// [PROBE] Runs from VulkanBasicExecutionDirect::onExecute after the queue has
// been flushed: the dispatch and the probe copy are guaranteed complete.
void VulkanFuse::afterExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // [ISP-IN] per-op INPUT dump: what the shader's binding-1 actually saw.
    // Buffer captured NOW (execute time) — dynamic buffers churn per frame.
    if (s_traceFile && mInProbeStage && !inputs.empty() && inputs[0] && inputs[0]->deviceId() != 0) {
        auto* vbn = static_cast<VulkanBackend*>(backend());
        auto tb = vbn->getBuffer(inputs[0]);
        VkBuffer src = std::get<0>(tb);
        size_t off = std::get<2>(tb);
        if (src != VK_NULL_HANDLE) {
            if (!mProbeCmd) mProbeCmd = vbn->getPool().allocBuffer();
            if (mProbeCmd) {
                VkBufferCopy bc{};
                bc.srcOffset = off;
                bc.dstOffset = 0;
                bc.size = 64;
                mProbeCmd->begin(0);
                vkCmdCopyBuffer(mProbeCmd->get(), src, mInProbeStage->buffer(), 1, &bc);
                mProbeCmd->end();
                vbn->pushCommand(mProbeCmd->get());
                vbn->finish();
                const float* pv = (const float*)mInProbeStage->map(0, 64);
                if (pv) {
                    fprintf(s_traceFile, "[ISP-IN] %s off=%zu f[0..7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        mType.c_str(), off, pv[0],pv[1],pv[2],pv[3],pv[4],pv[5],pv[6],pv[7]);
                    fflush(s_traceFile);
                    mInProbeStage->unmap();
                }
            }
        }
    }
    // [ISP-OUT] per-op output dump: read the FIRST output's first 16 floats
    // straight from the work buffer the shader actually wrote (not the
    // session's orphaned readback tensor). Buffer captured NOW (execute time).
    if (s_traceFile && mProbeStage && !outputs.empty() && outputs[0] && outputs[0]->deviceId() != 0) {
        auto* vbn = static_cast<VulkanBackend*>(backend());
        auto tb = vbn->getBuffer(outputs[0]);
        VkBuffer src = std::get<0>(tb);
        size_t off = std::get<2>(tb);
        if (src != VK_NULL_HANDLE) {
            if (!mProbeCmd) mProbeCmd = vbn->getPool().allocBuffer();
            if (mProbeCmd) {
                VkCommandBufferBeginInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(mProbeCmd->get(), &bi);
                VkBufferCopy bc2{};
                bc2.srcOffset = off;
                bc2.dstOffset = 0;
                bc2.size = 64;
                vkCmdCopyBuffer(mProbeCmd->get(), src, mProbeStage->buffer(), 1, &bc2);
                // [CONSTVERIFY] also read binding-0 (params SSBO) as the device sees it
                if (mConstStorageBuffer) {
                    VkBufferCopy bc3{};
                    bc3.srcOffset = 0;
                    bc3.dstOffset = 64;
                    bc3.size = 32;
                    vkCmdCopyBuffer(mProbeCmd->get(), mConstStorageBuffer->buffer(), mProbeStage->buffer(), 1, &bc3);
                }
                // also read the high region (where FRAMEFILL wrote)
                size_t outSize = 0;
                {
                    auto ob2 = vbn->getBuffer(outputs[0]);
                    outSize = std::get<1>(ob2);
                    size_t hi2 = outSize > 128 ? outSize - 64 : 0;
                    if (hi2 > 0) {
                        VkBufferCopy bc4{};
                        bc4.srcOffset = std::get<2>(ob2) + hi2;
                        bc4.dstOffset = 96;
                        bc4.size = 64;
                        vkCmdCopyBuffer(mProbeCmd->get(), src, mProbeStage->buffer(), 1, &bc4);
                    }
                }
                mProbeCmd->end();
                vbn->pushCommand(mProbeCmd->get());
                vbn->finish();
                const float* pv = (const float*)mProbeStage->map(0, 160);
                if (pv) {
                    fprintf(s_traceFile, "[OUTPROBE] %s off=%zu out[0..7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f consts=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f] hi=[%.4f,%.4f,%.4f,%.4f]\n",
                        mType.c_str(), off,
                        pv[0],pv[1],pv[2],pv[3],pv[4],pv[5],pv[6],pv[7],
                        pv[16],pv[17],pv[18],pv[19],pv[20],pv[21],pv[22],pv[23],
                        pv[24],pv[25],pv[26],pv[27]);
                    fflush(s_traceFile);
                    mProbeStage->unmap();
                }
            }
        }
    }
    if (mInProbeStage) {
        const float* ip = (const float*)mInProbeStage->map(0, 196608);
        if (ip) {
            fprintf(stderr, "[VulkanFuse-IN] off=%zu f[0..11]=", mInProbeOff);
            for (int i = 0; i < 12; ++i) fprintf(stderr, "%.6f,", ip[i]);
            // Boundary samples on the INPUT buffer: dense or zero past
            // f=12288? Decides whether the feed itself is truncated.
            fprintf(stderr, "\n[VulkanFuse-IN-BND] i12284..91=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f i38416..21=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f i45052..59=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                ip[12284],ip[12285],ip[12286],ip[12287],ip[12288],ip[12289],ip[12290],ip[12291],
                ip[38416],ip[38417],ip[38418],ip[38419],ip[38420],ip[38421],
                ip[45052],ip[45053],ip[45054],ip[45055],ip[45056],ip[45057],ip[45058],ip[45059]);
            fflush(stderr);
            mInProbeStage->unmap();
        }
    }
    if (!mProbeStage) {
        return;
    }
    const float* sp = (const float*)mProbeStage->map(0, 196608);
    if (sp) {
        fprintf(stderr, "[VulkanFuse-PROBE] type=%s off=%zu\n", mType.c_str(), mProbeOff);
        fprintf(stderr, "[VulkanFuse-PROBE-W0] ");
        for (int i = 0; i < 96; ++i) fprintf(stderr, "%.6f,", sp[i]);
        // Boundary probes: is the device buffer dense past f=12288 (the
        // suspected 49152B cut) and in the high region?
        fprintf(stderr, "\n[VulkanFuse-PROBE-BND] f12284..91=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f f38416..21=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f f45052..59=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
            sp[12284],sp[12285],sp[12286],sp[12287],sp[12288],sp[12289],sp[12290],sp[12291],
            sp[38416],sp[38417],sp[38418],sp[38419],sp[38420],sp[38421],
            sp[45052],sp[45053],sp[45054],sp[45055],sp[45056],sp[45057],sp[45058],sp[45059]);
        fprintf(stderr, "\n[VulkanFuse-PROBE-W1] ");
        for (int i = 88; i < 104; ++i) fprintf(stderr, "%.6f,", sp[i]);
        fprintf(stderr, "\n[VulkanFuse-PROBE-W2] ");
        for (int i = 352; i < 384; ++i) fprintf(stderr, "%.6f,", sp[i]);
        fprintf(stderr, "\n");
        fflush(stderr);
        mProbeStage->unmap();
    }
}

} // namespace MNN

// Register VulkanFuseCreator for OpType_Extra
extern "C" __attribute__((visibility("default"))) void MNNVulkanFuseRegister() {
    MNN::VulkanBackend::addCreator(MNN::OpType_Extra, new MNN::VulkanFuseCreator);
}
