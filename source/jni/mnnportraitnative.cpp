//
//  mnnportraitnative.cpp
//  MNN
//
//  Created by MNN on 2019/01/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <android/bitmap.h>
#include <jni.h>
#include <string.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>

extern "C" JNIEXPORT jintArray JNICALL
Java_com_taobao_android_mnn_MNNPortraitNative_nativeConvertMaskToPixelsMultiChannels(JNIEnv *env, jclass jclazz,
                                                                                     jfloatArray jmaskarray,
                                                                                     jint length) {
    float *scores = (float *)env->GetFloatArrayElements(jmaskarray, 0);

    int dst32[length];

#if 0
    for (int l = 0; l < length; l++) {
        int* dst = dst32 + l;
        float* src = scores + l;
        float max = scores[l];
        float min = scores[l];
        for(int c = 0; c < 21; c++){
            float data = src[c*length];
            if(max < data){
                max = data;
            }
            if(min > data){
                min = data;
            }
        }

        unsigned data = src[15*length];
        float range = 255.0f / (exp(max) - exp(min));
        float result = (exp(data) - exp(min)) * range;

        unsigned result_uint8 = result > 255.0f ? 255 : result;

        unsigned a = result_uint8;
        unsigned r = a;
        unsigned g = a;
        unsigned b = a;
        // ARGB
        dst[0] = a << 24 | r << 16 | g << 8 | b;
    }
#else
    for (int l = 0; l < length; l++) {
        int *dst   = dst32 + l;
        float *src = scores + l;
        float max  = scores[l];
        for (int c = 0; c < 21; c++) {
            if (max < src[c * length]) {
                max = src[c * length];
            }
        }
        unsigned a = src[15 * length] == max ? 0 : 255;
        unsigned r = a;
        unsigned g = a;
        unsigned b = a;
        // ARGB
        dst[0] = a << 24 | r << 16 | g << 8 | b;
    }
#endif
    jintArray arr = env->NewIntArray(length);
    env->SetIntArrayRegion(arr, 0, length, dst32);

    env->ReleaseFloatArrayElements(jmaskarray, scores, 0);

    return arr;
}

extern "C" JNIEXPORT jint JNICALL Java_com_taobao_android_mnn_MNNPortraitNative_nativeTensorGetData(JNIEnv *env, jclass type,
                                                                                                     jlong tensorPtr,
                                                                                                     jfloatArray dest) {
    auto tensor = reinterpret_cast<MNN::Tensor *>(tensorPtr);
    if (nullptr == dest) {
        std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), false));
        return hostTensor->elementSize();
    }
    auto length = env->GetArrayLength(dest);
    std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), true));
    tensor->copyToHostTensor(hostTensor.get());
    tensor = hostTensor.get();

    auto size = tensor->elementSize();
    if (length < size) {
        MNN_ERROR("Can't copy buffer, length no enough");
        return JNI_FALSE;
    }
    auto destPtr = env->GetFloatArrayElements(dest, nullptr);
    ::memcpy(destPtr, tensor->host<float>(), size * sizeof(float));
    env->ReleaseFloatArrayElements(dest, destPtr, 0);

    return JNI_TRUE;
}

