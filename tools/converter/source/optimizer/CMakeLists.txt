file(GLOB_RECURSE OPTIMIZER_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
# Per-format extra optimizer passes are compiled only when the corresponding
# converter backend is enabled; this keeps the existing code present in the
# tree without pulling in deps of disabled backends.
if(NOT MNN_BUILD_CONVERTER_TFLITE)
    file(GLOB TFLITEXTRA_SRC ${CMAKE_CURRENT_LIST_DIR}/tflitextra/*.cpp)
    list(REMOVE_ITEM OPTIMIZER_SRC ${TFLITEXTRA_SRC})
endif()
if(NOT MNN_BUILD_CONVERTER_TENSORFLOW)
    file(GLOB TFEXTRA_SRC ${CMAKE_CURRENT_LIST_DIR}/tfextra/*.cpp)
    list(REMOVE_ITEM OPTIMIZER_SRC ${TFEXTRA_SRC})
endif()
if(NOT MNN_BUILD_CONVERTER_CAFFE)
    file(GLOB CAFFEEXTRA_SRC ${CMAKE_CURRENT_LIST_DIR}/caffeextra/*.cpp)
    list(REMOVE_ITEM OPTIMIZER_SRC ${CAFFEEXTRA_SRC})
endif()
if(NOT MNN_BUILD_TORCH)
    file(GLOB TORCHEXTRA_SRC ${CMAKE_CURRENT_LIST_DIR}/torchextra/*.cpp)
    list(REMOVE_ITEM OPTIMIZER_SRC ${TORCHEXTRA_SRC})
endif()
add_library(MNNConverterOpt OBJECT ${OPTIMIZER_SRC})
# MNN_compression.ph.h
add_dependencies(MNNConverterOpt MNNCompress)
target_include_directories(MNNConverterOpt PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/)
# Embed ISP SPIR-V shaders (isp.unpack_blc, isp.demosaic_ccm, ...) so the
# IspChainFusion pass can produce fused VulkanFuse Extra ops. Controlled by
# MNN_ISP_EMBED_SPIRV (default ON in top-level CMakeLists.txt).
if(MNN_ISP_EMBED_SPIRV)
    target_compile_definitions(MNNConverterOpt PRIVATE MNN_ISP_EMBED_SPIRV)
endif()
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterOpt>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterOpt)

# [ISP_FUSION_DISABLED] Per user directive 2026-08-14: disable IspChainFusion pass1/pass2.
# Only pass0 (ONNX→MNN primitives + direct-emit isp.* Exos) should run.
# IspChainFusion produces isp.* Extra ops WITHOUT SPIR-V → VulkanFuse rejects →
# silent CPU fallback → undetectable. Disabling ensures direct-emit is the only
# path for isp.* ops.
if(ISP_FUSION_DISABLED)
    target_compile_definitions(MNNConverterOpt PRIVATE ISP_FUSION_DISABLED)
endif()
