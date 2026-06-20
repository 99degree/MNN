
file(GLOB MNN_BF16_SRCS "${CMAKE_CURRENT_LIST_DIR}/*")

file(GLOB MNN_BF16_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/*")

add_library(
    MNN_BF16
    STATIC
    ${MNN_BF16_SRCS}
    )
mnn_apply_build_flags(MNN_BF16)
target_compile_options(MNN_BF16 PRIVATE -DMNN_SUPPORT_BF16)
