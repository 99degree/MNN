//
//  VulkanBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBackend.hpp"
#include <algorithm>
#include "core/Execution.hpp"
#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "component/VulkanDevice.hpp"
#include "component/VulkanInstance.hpp"
#include "execution/VulkanBasicExecution.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <mutex>
#include <vector>
#include <cstring>

// ── Path-agnostic Vulkan-backend profiler registry ───────────
// The VulkanTimeProfiler lives on each VulkanBackend instance. Depending on
// the execution path the active backend may not be reachable via
// Interpreter::getBackend(session, tensor):
//   * Session path  (pass0 primitives) -> runSession(sess); backend reachable
//       via getBackend(sess, inputTensor).
//   * Module path  (direct-emit isp.* / VulkanFuse) -> Module::onForward;
//       the Session is never run, so getBackend returns a stale CPU backend.
// To support BOTH, every VulkanBackend registers itself here on construction;
// the dumpProfile() bridge scans for the first backend with live samples.
namespace {
std::mutex g_vkProfMutex;
std::vector<const MNN::VulkanBackend*> g_vkProfBackends;
void _regVkProf(const MNN::VulkanBackend* vb) {
    std::lock_guard<std::mutex> lk(g_vkProfMutex);
    g_vkProfBackends.push_back(vb);
}
void _unregVkProf(const MNN::VulkanBackend* vb) {
    std::lock_guard<std::mutex> lk(g_vkProfMutex);
    g_vkProfBackends.erase(
        std::remove(g_vkProfBackends.begin(), g_vkProfBackends.end(), vb),
        g_vkProfBackends.end());
}
// Returns the first backend whose profiler has samples (mNext>0).
const char* _dumpVkProf() {
    std::lock_guard<std::mutex> lk(g_vkProfMutex);
    for (const MNN::VulkanBackend* vb : g_vkProfBackends) {
        if (!vb) continue;
        std::string s = vb->getProfileString();  // cheap "" return if mNext==0
        if (!s.empty()) {
            char* c = (char*)malloc(s.size() + 1);
            if (c) memcpy(c, s.c_str(), s.size() + 1);
            return c;
        }
    }
    return strdup("ERR:no_active_vulkan_profile");
}
}
// #define MNN_OP_SUPPORT_LOG

#ifdef ENABLE_VULKAN_TIME_PROFILE
#include <chrono>
#endif
//#define MNN_VULKAN_DUMP_MEMORY_USAGE


namespace MNN {

static std::map<OpType, VulkanBackend::Creator*>* gCreator = nullptr;

// Creator
static inline std::map<OpType, VulkanBackend::Creator*>* getCreatorMap() {
    if (nullptr == gCreator) {
        gCreator = new std::map<OpType, VulkanBackend::Creator*>();
    }
    return gCreator;
}

template<typename T0, typename T1>
void _copy(const T0* src, T1* dst, size_t size) {
    for (int i=0; i<size; ++i) {
        dst[i] = src[i];
    }
}

#ifndef MNN_USE_ARMV82

void _VKFloatToHalf(const float* src, int16_t* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        ((half_float::half *)dst)[i] = (half_float::half)(src[i]);
    }
    return;
}

void _VKHalfToFloat(const int16_t* src, float* dst, size_t size) {
    const size_t batchSize = 8;
    std::vector<half_float::half> halfBatch(batchSize);

    for (size_t i = 0; i < size; i += batchSize) {
        size_t currentBatchSize = std::min(batchSize, size - i);

        ::memcpy(halfBatch.data(), &(src[i]), currentBatchSize * sizeof(int16_t));

        for (size_t j = 0; j < currentBatchSize; ++j) {
            dst[i + j] = static_cast<float>(halfBatch[j]);
        }
    }

    return;
}

#endif

static void _copyBufferToTensor(const Tensor* dest, const VulkanBuffer* source, size_t offset, bool half2float = false) {
    auto sourcePtr   = (const float*)source->map(offset);
    if (half2float) {
        auto dstPtr = dest->host<float>();
        auto elementCount = static_cast<size_t>(dest->elementSize());
        HALF_TO_FLOAT(reinterpret_cast<const int16_t*>(sourcePtr), dstPtr, elementCount);
    } else {
        ::memcpy(dest->host<float>(), sourcePtr, dest->usize());
    }
    source->unmap();
}

static void _copyTensorToBuffer(const Tensor* source, const VulkanBuffer* dest, size_t offset, bool float2half = false) {
    auto destPtr = reinterpret_cast<uint8_t*>(dest->map(offset));
    if (float2half) {
        auto srcPtr = source->host<float>();
        auto elementCount = static_cast<size_t>(source->elementSize());
        FLOAT_TO_HALF(srcPtr, reinterpret_cast<int16_t*>(destPtr), elementCount);
    } else {
        ::memcpy(destPtr, source->host<float>(), source->usize());
    }
    dest->unmap();
}

VulkanBackend::VulkanBackend(const VulkanRuntime* runtime) : Backend(MNN_FORWARD_VULKAN) {
    mRuntime = runtime;
    mDirect = (mRuntime->mGpuMode & MNNGpuMode::MNN_GPU_RECORD_BATCH) == 0;
    mUseFP16 = (mRuntime->mPrecision != BackendConfig::Precision_High && mRuntime->mDevice->getFP16Support());
    std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(runtime->mBufferPool.get());
    mDynamicBufferPool.resize(2);
    mDynamicBufferPool[0].reset(new EagerBufferAllocator(allocReal, mRuntime->mDevice->proty().limits.nonCoherentAtomSize));
    mCurrentDynamicBufferPool = mDynamicBufferPool[0].get();

    auto& dev              = device();
    mFence                 = std::make_shared<VulkanFence>(dev);
#ifdef ENABLE_VULKAN_TIME_PROFILE
    mTimeProfiler = std::make_shared<VulkanTimeProfiler>(dev);
#endif
#ifdef ENABLE_VULKAN_TIME_PROFILE
    _regVkProf(this);
#endif
    std::string deviceName = dev.proty().deviceName;
    if(deviceName.find("Apple") != std::string::npos){
        mUseAutoTune = false;
    }
    mCmdBufferForCopy.reset(runtime->mCmdPool->allocBuffer());
    // Small valid fallback buffer so VulkanFuse never binds VK_NULL_HANDLE
    // (which Mesa/freedreno treats as a null deref). See mVulkanDummyBuffer.
    if (!mVulkanDummyBuffer) {
        mVulkanDummyBuffer = std::make_shared<VulkanBuffer>(
            getMemoryPool(), false, 4096, nullptr,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

VulkanBackend::~VulkanBackend() {
#ifdef ENABLE_VULKAN_TIME_PROFILE
    _unregVkProf(this);
#endif
    /*keep release order*/
    mCurrentIndirectSegment = nullptr;
    mIndirectSegments.clear();
    mCmdBuffers.clear();
    mFence = nullptr;
}
void VulkanBackend::pushCommand(VkCommandBuffer buffer) const {
    mCmdBuffers.emplace_back(buffer);
}

std::shared_ptr<VulkanCommandPool::Buffer> VulkanBackend::acquireIndirectSegmentForRecord() {
    MNN_ASSERT(!mDirect);
    if (nullptr == mCurrentIndirectSegment.get()) {
        mCurrentIndirectSegment.reset(mRuntime->mCmdPool->allocBuffer());
        mCurrentIndirectSegment->begin(0);
        mCurrentIndirectSegmentOpCount = 0;
    }
    return mCurrentIndirectSegment;
}

void VulkanBackend::finishIndirectRecordedOp() {
    if (mDirect) {
        return;
    }
    MNN_ASSERT(nullptr != mCurrentIndirectSegment.get());
    ++mCurrentIndirectSegmentOpCount;
    if (mCurrentIndirectSegmentOpCount >= kIndirectSegmentOpLimit) {
        _sealIndirectSegment();
    }
}

const VulkanPipeline* VulkanBackend::getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                                 const std::vector<uint32_t>& localSize,
                                                 const std::vector<uint32_t>& specConstants) const {
    return mRuntime->mPipelineFactory->getPipeline(key, types, localSize, specConstants);
}

SharedPtr<VulkanPipeline> VulkanBackend::getPrivatePipeline(const std::string& key, const std::vector<VkDescriptorType>& types, const std::vector<uint32_t>& specConstants) {
    return mRuntime->mPipelineFactory->getPrivatePipeline(key, types, specConstants);
}

void VulkanBackend::onResizeBegin() {
#ifdef ENABLE_VULKAN_TIME_PROFILE
    if (mTimeProfiler) {
        mTimeProfiler->reset();
    }
#endif
    if (!mDirect) {
        _resetIndirectSegments();
    }
}
ErrorCode VulkanBackend::onResizeEnd() {
    if (!mDirect) {
        _sealIndirectSegment();
    }
    mHostBuffer.reset();
    return NO_ERROR;
}
class VulkanMemRelease : public Backend::MemObj {
public:
    VulkanMemRelease(BufferAllocator* allocator, MemChunk points, int size) {
        mPoint = std::move(points);
        mAllocator = allocator;
        mSize = size;
    }
    virtual ~ VulkanMemRelease() {
        mAllocator->free(mPoint);
    }
    inline int getSize() const {
        return mSize;
    }
    inline MemChunk points() const {
        return mPoint;
    }
private:
    BufferAllocator* mAllocator;
    MemChunk mPoint;
    int mSize;
};
VULKAN_TENSOR VulkanBackend::getBuffer(const Tensor* tensor) const {
    // A tensor that is not resident on this Vulkan device (e.g. a Host/const
    // tensor reaching a VulkanFuse op, or a tensor whose deviceId was never
    // allocated on the GPU) must NOT yield a null VkBuffer: binding
    // VK_NULL_HANDLE to a descriptor set is invalid and crashes Mesa/freedreno.
    if (!tensor || tensor->deviceId() == 0) {
        size_t sz = tensor ? getTensorSize(tensor) : 0;
        VkBuffer buf = (mVulkanDummyBuffer && mVulkanDummyBuffer->buffer())
            ? mVulkanDummyBuffer->buffer() : VK_NULL_HANDLE;
        return std::make_tuple(buf, sz, 0);
    }
    auto b = getTensorBuffer(tensor);
    return std::make_tuple(b.first->buffer(), getTensorSize(tensor), b.second);
}

std::pair<const VulkanBuffer*, size_t> VulkanBackend::getTensorBuffer(const Tensor* tensor) const {
    auto mem = (VulkanBuffer*)(tensor->deviceId());
    MNN_ASSERT(nullptr != mem);
    return std::make_pair(mem, TensorUtils::getDescribeOrigin(tensor)->offset);
}

size_t VulkanBackend::getTensorSize(const Tensor* tensor) const {
    size_t alignElementSize = (size_t) UP_DIV(tensor->elementSize(), 4) * 4;
    size_t bytes = ((tensor->getType().code == halide_type_float) && mUseFP16) ? sizeof(uint16_t) : sizeof(float);
    size_t size = alignElementSize * bytes;
    return size;
}

Backend::MemObj* VulkanBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    MNN_ASSERT(tensor->getType().code == halide_type_float || tensor->getType().code == halide_type_int);
    //FUNC_PRINT_ALL(tensor, p);
    auto alignSize = getTensorSize(tensor);
    auto MTensor     = const_cast<Tensor*>(tensor);
    auto des = TensorUtils::getDescribeOrigin(tensor);
    if (Backend::STATIC == storageType) {
        auto newBuffer = mRuntime->mBufferPool->alloc(alignSize);
        if (nullptr == newBuffer.first) {
            MNN_ERROR("Vulkan alloc static buffer failed, size=%zu\n", alignSize);
            return nullptr;
        }
        auto mem = new VulkanMemRelease(mRuntime->mBufferPool.get(), newBuffer, alignSize);
        MTensor->buffer().device = (uint64_t)(newBuffer.first);
        des->offset = newBuffer.second;
        return mem;
    }
    bool seperate  = storageType == Backend::DYNAMIC_SEPERATE;
    auto newBuffer = mCurrentDynamicBufferPool->alloc(alignSize, seperate);
    if (nullptr == newBuffer.first) {
        MNN_ERROR("Vulkan alloc dynamic buffer failed, size=%zu\n", alignSize);
        return nullptr;
    }
    auto mem = new VulkanMemRelease(mCurrentDynamicBufferPool, newBuffer, alignSize);
    MTensor->buffer().device = (uint64_t)(newBuffer.first);
    des->offset = newBuffer.second;
    return mem;
}
bool VulkanBackend::onSelectDynamicAllocator(int index, int maxIndex) {
    if (maxIndex > 2 || index >= 2 || index < 0) {
        return false;
    }
    if (mDynamicBufferPool[1].get() == nullptr) {
        std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(mRuntime->mBufferPool.get());
        mDynamicBufferPool[1].reset(new EagerBufferAllocator(allocReal, mRuntime->mDevice->proty().limits.nonCoherentAtomSize));
    }
    mCurrentDynamicBufferPool = mDynamicBufferPool[index].get();
    return true;
}

std::shared_ptr<VulkanBuffer> VulkanBackend::allocUniform(const void* src, int size) {
    auto rt = const_cast<VulkanRuntime*>(mRuntime);
    return rt->allocUniform(src, size);
}
void VulkanBackend::recycleUniform(std::shared_ptr<VulkanBuffer> buffer) {
    auto rt = const_cast<VulkanRuntime*>(mRuntime);
    rt->recycleUniform(buffer);
}

bool VulkanBackend::onClearBuffer() {
    mCurrentDynamicBufferPool->release(false);
    _resetIndirectSegments();
    return true;
}
Execution* VulkanBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
    auto creator = getCreatorMap();
    auto iter    = creator->find(op->type());
    std::string name = "";
    if (nullptr != op->name()) {
        name = op->name()->str();
    }
    if (iter == creator->end()) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_PRINT("Vulkan don't support %d, %s: %s\n", op->type(), EnumNameOpType(op->type()),
                name.c_str());
#endif
        return nullptr;
    }
    std::shared_ptr<VulkanBasicExecution> originExecution ((VulkanBasicExecution*)iter->second->onCreate(inputs, outputs, op, this));
    if (nullptr == originExecution) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_ERROR("Vulkan don't support for %s, type=%s, Special case\n", name.c_str(), EnumNameOpType(op->type()));
#endif
        return nullptr;
    }

#ifdef ENABLE_VULKAN_TIME_PROFILE
    if (op->type() == OpType_Extra) {
        // Extra ops (incl. direct-emit isp.*) are already named in their
        // VulkanFuse ctor by the real isp.* type string, so the per-op GPU
        // timing profile shows each isp.* stage instead of one "Extra" bucket.
        // Preserve that name here.
    } else {
        originExecution->setName(EnumNameOpType(op->type()));
    }
#endif

    if (mDirect) {
        return new VulkanBasicExecutionDirect(originExecution);
    }
    return new VulkanBasicExecutionInDirect(originExecution);
}

void VulkanBackend::onExecuteBegin() const {
}

void VulkanBackend::onExecuteEnd() const {
    if (!mDirect) {
        mCmdBuffers.reserve(mCmdBuffers.size() + mIndirectSegments.size());
        for (auto& segment : mIndirectSegments) {
            mCmdBuffers.push_back(segment->get());
        }
    }
#ifdef ENABLE_VULKAN_TIME_PROFILE
    auto startTime = std::chrono::high_resolution_clock::now();
    _finish();
    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / (1e6f);
    if (mTimeProfiler) {
        // Store Execution-level GPU time so callers can query it via
        // Runtime::onGetLastGpuTimeMs() without parsing printed output.
        mRuntime->mLastGpuTimeMs = mTimeProfiler->getTotalTime(VulkanTimeProfiler::Kind::Execution);

#ifndef MNN_GPU_PROFILE_SILENT
        MNN_PRINT("\n=============== Vulkan Time Profiling (Begin) ===============\n");
        mTimeProfiler->printTimeProfile();
        MNN_PRINT("Total time calculated by CPU is %6.2f ms.\n", totalTime);
        MNN_PRINT("\n================ Vulkan Time Profiling (End) ================\n");
#endif
    }
#else
    _finish();
#endif
}

void VulkanBackend::finish() {
    _finish();
}

void VulkanBackend::_finish() const {
    if (mCmdBuffers.empty()) {
        return;
    }
    VkSubmitInfo submit_info = {/* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                /* .pNext                = */ nullptr,
                                /* .waitSemaphoreCount   = */ 0,
                                /* .pWaitSemaphores      = */ nullptr,
                                /* .pWaitDstStageMask    = */ nullptr,
                                /* .commandBufferCount   = */ (uint32_t)mCmdBuffers.size(),
                                /* .pCommandBuffers      = */ mCmdBuffers.data(),
                                /* .signalSemaphoreCount = */ 0,
                                /* .pSignalSemaphores    = */ nullptr};
    auto fenceReal           = mFence->get();
    mFence->reset();
    CALL_VK(vkQueueSubmit(device().acquireDefaultDevQueue(), 1, &submit_info, fenceReal));

    auto res = mFence->wait();
    MNN_VK_CHECK(res);
    mCmdBuffers.clear();
}

void VulkanBackend::_resetIndirectSegments() {
    mCurrentIndirectSegment = nullptr;
    mIndirectSegments.clear();
    mCurrentIndirectSegmentOpCount = 0;
}

void VulkanBackend::_sealIndirectSegment() {
    if (nullptr == mCurrentIndirectSegment.get()) {
        return;
    }
    if (mCurrentIndirectSegmentOpCount <= 0) {
        mCurrentIndirectSegment = nullptr;
        return;
    }
    mCurrentIndirectSegment->end();
    mIndirectSegments.emplace_back(mCurrentIndirectSegment);
    mCurrentIndirectSegment = nullptr;
    mCurrentIndirectSegmentOpCount = 0;
}

const VulkanDevice& VulkanBackend::device() const {
    return (* mRuntime->mDevice);
}
const VulkanPipelineFactory* VulkanBackend::getPipelineFactory() const {
    return mRuntime->mPipelineFactory.get();
}

static Tensor::DimensionType _convert(MNN_DATA_FORMAT format) {
    switch (format) {
        case MNN_DATA_FORMAT_NCHW:
            return Tensor::CAFFE;
        case MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        case MNN_DATA_FORMAT_NHWC:
            return Tensor::TENSORFLOW;
        default:
            break;
    }
    return Tensor::CAFFE;
}
std::shared_ptr<VulkanBuffer> VulkanBackend::createHostBuffer(size_t size) const {
    std::shared_ptr<VulkanBuffer> res;
    res.reset(new VulkanBuffer(*mRuntime->mMemoryPool, false, size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    return res;
}

void VulkanBackend::copyGPUToGPUBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) const {
    auto cmdbuffer = mCmdBufferForCopy;
    cmdbuffer->begin(0);
    VkBufferCopy bufferCopy;
    bufferCopy.size = size;
    bufferCopy.dstOffset = dstOffset;
    bufferCopy.srcOffset = srcOffset;
    vkCmdCopyBuffer(cmdbuffer->get(), srcBuffer, dstBuffer,
                    1, &bufferCopy);
    cmdbuffer->end();
    pushCommand(cmdbuffer->get());
    _finish();
}

void VulkanBackend::copyGPUToGPUBufferRegions(VkBuffer srcBuffer, VkBuffer dstBuffer, const VkBufferCopy* regions,
                                              uint32_t regionCount) const {
    if (regionCount == 0 || regions == nullptr) {
        return;
    }
    auto cmdbuffer = mCmdBufferForCopy;
    cmdbuffer->begin(0);
    vkCmdCopyBuffer(cmdbuffer->get(), srcBuffer, dstBuffer, regionCount, regions);
    cmdbuffer->end();
    pushCommand(cmdbuffer->get());
    _finish();
}

void VulkanBackend::copyToGPUBuffer(const void* src, VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset) const {
    _requireHostBuffer(size);
    ::memcpy(mHostBuffer->map(), src, size);
    mHostBuffer->unmap();
    copyGPUToGPUBuffer(mHostBuffer->buffer(), buffer, size, 0, offset);
}
void VulkanBackend::_requireHostBuffer(size_t size) const {
    _finish();
    if (nullptr == mHostBuffer || mHostBuffer->size() < size) {
        mHostBuffer = createHostBuffer(size);
    }
}

void VulkanBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef MNN_VULKAN_DEBUG
    AUTOTIME;
    MNN_PRINT("Src: ");
    for (int i=0; i<srcTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", srcTensor->length(i));
    }
    MNN_PRINT("\n");
    MNN_PRINT("Dst: ");
    for (int i=0; i<dstTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", dstTensor->length(i));
    }
    MNN_PRINT("\n");
#endif

    auto calculateCpSize = [this] (const Tensor* tensor) -> size_t {
            size_t eleSize = (size_t) tensor->elementSize();
            return (tensor->getType().code == halide_type_float && this->mUseFP16) ?
                (eleSize * sizeof(uint16_t)) :
                (eleSize * sizeof(float));
        };

    std::shared_ptr<Tensor> tempTensor;
    if (srcTensor->host<float>() != nullptr) {
        _finish();
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        auto buffer = reinterpret_cast<VulkanBuffer*>(dstTensor->deviceId());
        auto offset = TensorUtils::getDescribeOrigin(dstTensor)->offset;
        // host->gpu
        if(format != TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            tempTensor.reset(Tensor::create(dstTensor->shape(), dstTensor->getType(), nullptr, _convert(format)));
            MNNCPUCopyBuffer(srcTensor, tempTensor.get());
            srcTensor = tempTensor.get();
        }
        size_t cpSize = calculateCpSize(srcTensor);
        _requireHostBuffer(cpSize);
        if (getenv("ISP_DEBUG_VLOG") && srcTensor->host<float>() != nullptr && cpSize >= 48) {
            const float* sv = srcTensor->host<float>();
            std::string dstr;
            for (int i = 0; i < srcTensor->dimensions(); ++i) {
                dstr += std::to_string(srcTensor->length(i)) + "x";
            }
            fprintf(stderr, "[HOST2GPU] dims=%s fmt=%d usize=%zu devOff=%zu host[0..11]=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                dstr.c_str(), (int)TensorUtils::getDescribe(srcTensor)->dimensionFormat,
                srcTensor->usize(), (size_t)offset,
                sv[0],sv[1],sv[2],sv[3],sv[4],sv[5],sv[6],sv[7],sv[8],sv[9],sv[10],sv[11]);
            fflush(stderr);
        }
        _copyTensorToBuffer(srcTensor, mHostBuffer.get(), 0, srcTensor->getType().code == halide_type_float && mUseFP16);
        if (getenv("ISP_DEBUG_VLOG") && cpSize >= 96) {
            const float* hv = (const float*)mHostBuffer->map(0, 96);
            if (hv) {
                fprintf(stderr, "[FEEDCOPY] cpSize=%zu devOffset=%zu h[0..11]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    cpSize, (size_t)offset, hv[0],hv[1],hv[2],hv[3],hv[4],hv[5],hv[6],hv[7],hv[8],hv[9],hv[10],hv[11]);
                fflush(stderr);
                mHostBuffer->unmap();
            }
        }
        auto cmdbuffer = mCmdBufferForCopy;
        cmdbuffer->begin(0);
        VkBufferCopy bufferCopy;
        bufferCopy.size = cpSize;
        bufferCopy.dstOffset = offset;
        bufferCopy.srcOffset = 0;
        vkCmdCopyBuffer(cmdbuffer->get(), mHostBuffer->buffer(), buffer->buffer(),
                        1, &bufferCopy);
        cmdbuffer->end();
        pushCommand(cmdbuffer->get());
        _finish();
    } else if (dstTensor->host<float>() != nullptr) {
        // gpu->host
        _finish();
        if (getenv("ISP_DEBUG_VLOG") && srcTensor->elementSize() >= 4096) {
            fprintf(stderr, "[GPU2HOST] srcFmt=%d dstFmt=%d dims=%zu uszSrc=%zu off=%zu\n",
                (int)TensorUtils::getDescribe(srcTensor)->dimensionFormat,
                (int)TensorUtils::getDescribe(dstTensor)->dimensionFormat,
                (size_t)srcTensor->dimensions(), srcTensor->usize(),
                (size_t)TensorUtils::getDescribeOrigin(srcTensor)->offset);
            fflush(stderr);
        }
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        if (format != TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            tempTensor.reset(Tensor::create(srcTensor->shape(), dstTensor->getType(), nullptr, _convert(TensorUtils::getDescribe(srcTensor)->dimensionFormat)), [dstTensor](void* t) {
                Tensor* temp = (Tensor*)t;
                MNNCPUCopyBuffer(temp, dstTensor);
                delete temp;
            });
            dstTensor = tempTensor.get();
        }
        size_t cpSize = calculateCpSize(dstTensor);
        _requireHostBuffer(cpSize);
        auto buffer = reinterpret_cast<VulkanBuffer*>(srcTensor->deviceId());
        auto offset = TensorUtils::getDescribeOrigin(srcTensor)->offset;
        auto cmdbuffer = mCmdBufferForCopy;
        cmdbuffer->begin(0);
        VkBufferCopy bufferCopy;
        bufferCopy.size = cpSize;
        bufferCopy.dstOffset = 0;
        bufferCopy.srcOffset = offset;
        vkCmdCopyBuffer(cmdbuffer->get(), buffer->buffer(), mHostBuffer->buffer(),
                        1, &bufferCopy);
        cmdbuffer->end();
        pushCommand(cmdbuffer->get());
        _finish();
        _copyBufferToTensor(dstTensor, mHostBuffer.get(), 0, dstTensor->getType().code == halide_type_float && mUseFP16);
        if (getenv("ISP_DEBUG_VLOG") && dstTensor->elementSize() >= 4096 && !mUseFP16) {
            const float* hv = (const float*)mHostBuffer->map(0, 196608);
            if (hv) {
                fprintf(stderr, "[GPU2HOST-STAGE] h[88..103]=%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                    hv[88],hv[89],hv[90],hv[91],hv[92],hv[93],hv[94],hv[95],
                    hv[96],hv[97],hv[98],hv[99],hv[100],hv[101],hv[102],hv[103]);
                fprintf(stderr, "[GPU2HOST-STAGE2] h[45056..45063]=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f h[38418..38421]=%.1f,%.1f,%.1f,%.1f h[49144..49151]=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                    hv[45056],hv[45057],hv[45058],hv[45059],hv[45060],hv[45061],hv[45062],hv[45063],
                    hv[38418],hv[38419],hv[38420],hv[38421],
                    hv[49144],hv[49145],hv[49146],hv[49147],hv[49148],hv[49149],hv[49150],hv[49151]);
                fflush(stderr);
                mHostBuffer->unmap();
            }
        }
    } else if (srcTensor->deviceId() != 0 && dstTensor->deviceId() != 0) {
        // gpu->gpu
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        MNN_ASSERT(format == TensorUtils::getDescribe(srcTensor)->dimensionFormat);
        std::shared_ptr<VulkanCommandPool::Buffer> buffer( mRuntime->mCmdPool->allocBuffer());
        buffer->begin(0);
        VkBufferCopy bufferCopy;
        size_t cpSize = calculateCpSize(srcTensor);
        bufferCopy.size = cpSize;
        bufferCopy.dstOffset = TensorUtils::getDescribeOrigin(dstTensor)->offset;
        bufferCopy.srcOffset = TensorUtils::getDescribeOrigin(srcTensor)->offset;
        vkCmdCopyBuffer(buffer->get(), reinterpret_cast<VulkanBuffer*>(srcTensor->deviceId())->buffer(), reinterpret_cast<VulkanBuffer*>(dstTensor->deviceId())->buffer(),
                        1, &bufferCopy);
        buffer->end();
        pushCommand(buffer->get());
        _finish();
    }
}
int VulkanBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    _finish();
    return 0;
}

bool VulkanBackend::addCreator(OpType t, Creator* c) {
    auto allKind = getCreatorMap();
    allKind->insert(std::make_pair(t, c));
    return true;
}

bool VulkanBackend::onGetTensorInfo(const Tensor* tensor, void* dstInfo) {
    if (nullptr == dstInfo) {
        return true;
    }
    auto vkBuffer = getBuffer(tensor);
    auto dst = (MNNVulkanTensorContent*)dstInfo;
    dst->buffer = std::get<0>(vkBuffer);
    dst->offset = std::get<2>(vkBuffer);
    dst->size = std::get<1>(vkBuffer);
    return true;
}

float VulkanBackend::getPipelineTime(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> groupSize){
    std::shared_ptr<VulkanCommandPool::Buffer> cmd;
    cmd.reset(const_cast<VulkanCommandPool::Buffer *>(getPool().allocBuffer()));
    cmd->begin(0);
    mRuntime->mQueryPool->VulkanCmdResetQueryPool(cmd.get()->get());
    mRuntime->mQueryPool->VulkanCmdWriteTimestamp(cmd.get()->get(), 0);
    pipeline->bind(cmd.get()->get(), des->get());
    vkCmdDispatch(cmd.get()->get(), groupSize[0], groupSize[1], groupSize[2]);
    mRuntime->mQueryPool->VulkanCmdWriteTimestamp(cmd.get()->get(), 1);
    cmd->end();
    getPool().submitAndWait(cmd.get()->get());
    float time = mRuntime->mQueryPool->VulkanGetQueryPoolResults();
    return time;
}

std::vector<uint32_t> VulkanBackend::autoTunePipeline(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> gws){
    std::vector<uint32_t> lws(3, 1);
    std::vector<int> groupSize(3, 1);
    std::vector<int> maxGroups(3, 1);
    int maxGroupSize = mRuntime->mDevice->getMaxComputeWorkGroupInvocations();
    mRuntime->mDevice->getMaxComputeWorkGroupSize(maxGroups);
    
    std::vector<uint32_t> lws_prefer(3, 1);
    float min_cost = -1.0f;
    
    while(lws[2] <= gws[2] && lws[2] <= maxGroups[2]) {
        lws[1] = 1;
        while(lws[1] <= gws[1] && lws[1] <= maxGroups[1]) {
            lws[0] = 1;
            while(lws[0] <= gws[0] && lws[0] <= maxGroups[0]) {
                if(lws[0]*lws[1]*lws[2] <= maxGroupSize) {
                    groupSize[0] = UP_DIV(gws[0], lws[0]);
                    groupSize[1] = UP_DIV(gws[1], lws[1]);
                    groupSize[2] = UP_DIV(gws[2], lws[2]);
                    
                    pipeline->changePipeline(lws);
                    auto cost_time = getPipelineTime(pipeline, des, groupSize);
                    if(cost_time < min_cost || min_cost < 0.0f) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                        lws_prefer[2] = lws[2];
                    }
                }
                lws[0]*=2;
            }
            lws[1]*=2;
        }
        lws[2]*=2;
    }
    
    pipeline->changePipeline(lws_prefer);
    return lws_prefer;
}


} // namespace MNN

/**
 * C-linkage bridge so non-MNN clients (the cam_app JNI in libmnnhelper.so)
 * can extract per-op (per-shader) GPU timing from the VulkanTimeProfiler
 * WITHOUT needing the internal VulkanBackend.hpp header.
 *
 * Pass a `const MNN::Backend*` obtained from
 * `Interpreter::getBackend(session, outputTensor)`; this function
 * dynamic_casts it to `VulkanBackend*` and returns the aggregated per-op
 * profile string. Returns nullptr if the backend is not Vulkan or profiling
 * is unavailable (caller must free with free() if non-null).
 */
;// MNN Vulkan builds with -fvisibility=hidden, so mark the export.
// Registry-based per-op GPU timing export. No backend pointer needed:
// scans all live VulkanBackend instances (covers BOTH Session and Module
// execution paths) and returns the first with active profiling samples.
// Caller frees the returned string with free().
#pragma GCC visibility push(default)
extern "C" const char* mnn_vulkan_backend_dumpProfile() {
#ifdef ENABLE_VULKAN_TIME_PROFILE
    const char* r = _dumpVkProf();
    return r;
#else
    return strdup("ERR:profile_not_built");
#endif
}
#pragma GCC visibility pop
