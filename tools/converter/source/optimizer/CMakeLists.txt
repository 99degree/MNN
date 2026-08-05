file(GLOB_RECURSE OPTIMIZER_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
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
