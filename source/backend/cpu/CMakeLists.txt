# CPU
IF(MNN_CPU)
    if(MNN_SUPPORT_RENDER)
        FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/compute/* ${CMAKE_CURRENT_LIST_DIR}/render/*)
    elseif(MNN_MINIMAL_CPU)
        # Minimal CPU: core infrastructure only
        # Only CommonOptFunction.cpp is included from compute/ — it provides
        # MNNGetCoreFunctions(), NC4HW4 pack/unpack, binary/unary dispatch, etc.
        # All heavy compute ops (conv, pool, activation, etc.) are excluded.
        FILE(GLOB MNN_CPU_SRC
            ${CMAKE_CURRENT_LIST_DIR}/CPUBackend.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUBinary.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUCast.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUConcat.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUEltwise.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUExternalConst.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUMatMul.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPURaster.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPURelu.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUResizeCache.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPURuntime.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUSlice.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUScale.cpp
            ${CMAKE_CURRENT_LIST_DIR}/ThreadPool.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUSoftmax.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUTensorConvert.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUUnary.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUReshape.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUWhere.cpp
            ${CMAKE_CURRENT_LIST_DIR}/compute/CommonOptFunction.cpp
            ${CMAKE_CURRENT_LIST_DIR}/CPUOPRegisterMinimal.cpp
        )
    else()
        FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/compute/*)
        # Exclude CPUOPRegisterMinimal.cpp when building full CPU backend to avoid duplicate registerCPUOps()
        list(REMOVE_ITEM MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/CPUOPRegisterMinimal.cpp)
    endif()
    
    add_library(MNNCPU STATIC ${MNN_CPU_SRC})
    mnn_apply_build_flags(MNNCPU)
    
    if(MNN_MINIMAL_CPU)
        target_compile_definitions(MNNCPU PRIVATE MNN_MINIMAL_CPU=1)
        # When MNN_MINIMAL_CPU is set, the arm/NEON assembly files are excluded.
        # We must undefine MNN_USE_NEON so CommonOptFunction.cpp's scalar
        # fallbacks (guarded by #ifndef MNN_USE_NEON) are compiled.
        target_compile_options(MNNCPU PRIVATE -UMNN_USE_NEON)
    endif()
    
    if (MNN_PIPELINE_PROFILE)
        target_compile_options(MNNCPU PRIVATE -DMNN_PIPELINE_PROFILE)
    endif()
    
    if (MNN_SUPPORT_BF16)
        include(${CMAKE_CURRENT_LIST_DIR}/bf16/CMakeLists.txt)
        list(APPEND MNN_TARGETS MNN_BF16)
        list(APPEND MNN_STATIC_LIBS MNN_BF16)
        target_compile_options(MNNCPU PRIVATE -DMNN_SUPPORT_BF16)
    endif()
    
    list(APPEND MNN_STATIC_LIBS MNNCPU)
    list(APPEND MNN_TARGETS MNNCPU)
    
    if(MNN_USE_SPARSE_COMPUTE)
        target_compile_options(MNNCPU PRIVATE -DMNN_USE_SPARSE_COMPUTE)
    endif()
    
    # X86_64 AVX/SSE
    if (MNN_USE_SSE)
        include(${CMAKE_CURRENT_LIST_DIR}/x86_x64/CMakeLists.txt)
    endif()
    
    # AArch64 NEON assembly — include in minimal mode for core functions
    if(MNN_MINIMAL_CPU AND (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64"))
        message(STATUS "MNN_MINIMAL_CPU: including AArch64 NEON assembly")
        FILE(GLOB MNN_AArch64_SRC ${CMAKE_CURRENT_LIST_DIR}/arm/arm64/*.[sS])
        FILE(GLOB MNN_NEON_SRC ${CMAKE_CURRENT_LIST_DIR}/arm/CommonOptFunctionNeon.cpp)
        add_library(MNNARM64 STATIC ${MNN_AArch64_SRC} ${MNN_NEON_SRC})
        mnn_apply_build_flags(MNNARM64)
        target_include_directories(MNNARM64 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/arm/)
        target_compile_options(MNNARM64 PRIVATE -D__aarch64__)
        # MNN_USE_NEON is defined in source/core/Macro.h for aarch64, but we keep
        # it consistent by also passing it here
        target_compile_options(MNNARM64 PRIVATE -DMNN_USE_NEON)
        list(APPEND MNN_STATIC_LIBS MNNARM64)
        list(APPEND MNN_TARGETS MNNARM64)
        # Also add the definition to MNNCPU for consistency
        target_compile_options(MNNCPU PRIVATE -DMNN_USE_NEON)
    else()
        include(${CMAKE_CURRENT_LIST_DIR}/arm/CMakeLists.txt)
    endif()


    
    if(MNN_USE_RVV)
        include(${CMAKE_CURRENT_LIST_DIR}/riscv/CMakeLists.txt)
    endif()
    
    IF(NOT DEFINED IOS_ARCH) 
        set(IOS_ARCH "") 
    ENDIF()
    
    # ARM82 Assemblies — skip in minimal mode
    IF(MNN_ARM82 AND NOT MNN_MINIMAL_CPU)
        IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64")
            target_compile_options(MNNCPU PRIVATE -DENABLE_ARMV82)
            include(${CMAKE_CURRENT_LIST_DIR}/../arm82/CMakeLists.txt)
            list(APPEND MNN_TARGETS MNN_Arm82)
            list(APPEND MNN_STATIC_LIBS MNN_Arm82)
        ENDIF()
    ENDIF()
ENDIF()
