option(MNN_VULKAN_IMAGE "Use Image as basic type" ON)
if (MNN_VULKAN_IMAGE)
    FILE(GLOB_RECURSE MNN_Vulkan_SRC
        ${CMAKE_CURRENT_LIST_DIR}/image/*
        ${CMAKE_CURRENT_LIST_DIR}/component/*
        ${CMAKE_CURRENT_LIST_DIR}/runtime/*
        ${CMAKE_CURRENT_LIST_DIR}/vulkan/*
    )
else()
    FILE(GLOB_RECURSE MNN_Vulkan_SRC
        ${CMAKE_CURRENT_LIST_DIR}/buffer/*
        ${CMAKE_CURRENT_LIST_DIR}/component/*
        ${CMAKE_CURRENT_LIST_DIR}/runtime/*
        ${CMAKE_CURRENT_LIST_DIR}/vulkan/*
    )
endif()

# Exclude SPIR-V files and assembly (.asm/.spv) from the source list —
# they are data files included by C++ source, not compilable sources.
list(FILTER MNN_Vulkan_SRC EXCLUDE REGEX "\\.asm$")
list(FILTER MNN_Vulkan_SRC EXCLUDE REGEX "\\.spv$")
list(FILTER MNN_Vulkan_SRC EXCLUDE REGEX "\\.spv\\.h$")
# Include .spv.cpp files (they contain the SPIR-V binary data as C++ arrays)
# but exclude the .spv.cpp files from the buffer/shaders directory that we don't want
list(FILTER MNN_Vulkan_SRC EXCLUDE REGEX "buffer/shaders/.*\\.spv\\.cpp$")

if(MNN_GPU_TIME_PROFILE)
    add_definitions(-DENABLE_VULKAN_TIME_PROFILE)
    IF (MNN_GPU_PROFILE_SILENT)
        add_definitions(-DMNN_GPU_PROFILE_SILENT)
    ENDIF()
ENDIF()


include_directories("./")
if(MNN_USE_SYSTEM_LIB)
    find_package(Vulkan REQUIRED)
    # On Android the ONLY libvulkan that exists on a device is the system ICD
    # loader, whose SONAME is `libvulkan.so`. The host we build on can disagree:
    # Termux ships `libvulkan.so -> libvulkan.so.1`, so find_package() resolves
    # that and every artifact gets `NEEDED libvulkan.so.1` — which no Android
    # linker can satisfy (`dlopen failed: library "libvulkan.so.1" not found`).
    #
    # Prefer the NDK's sysroot stub: it is an aarch64 Android ELF with SONAME
    # `libvulkan.so`, so the NEEDED entry it produces is correct by
    # construction. Fall back to the discovered library only when no sysroot
    # stub exists (non-Android hosts).
    if (CMAKE_SYSTEM_NAME MATCHES "^Android" AND DEFINED ENV{ANDROID_NDK_ROOT})
        # Map the target ABI to the NDK sysroot's per-ABI library directory.
        # Getting this wrong is not a soft failure: linking an arm stub into an
        # aarch64 build is `ld.lld: error: ... is incompatible with
        # aarch64linux`.
        if (ANDROID_ABI STREQUAL "arm64-v8a")
            set(_mnn_ndk_triple "aarch64-linux-android")
        elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
            set(_mnn_ndk_triple "arm-linux-androideabi")
        elseif (ANDROID_ABI STREQUAL "x86_64")
            set(_mnn_ndk_triple "x86_64-linux-android")
        elseif (ANDROID_ABI STREQUAL "x86")
            set(_mnn_ndk_triple "i686-linux-android")
        else()
            set(_mnn_ndk_triple "aarch64-linux-android")
        endif()
        file(GLOB _mnn_ndk_vulkan_stubs
            "$ENV{ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/*/sysroot/usr/lib/${_mnn_ndk_triple}/*/libvulkan.so"
            )
        if (_mnn_ndk_vulkan_stubs)
            list(SORT _mnn_ndk_vulkan_stubs)
            list(GET _mnn_ndk_vulkan_stubs -1 MNN_VULKAN_ANDROID_STUB)
            set(Vulkan_LIBRARY "${MNN_VULKAN_ANDROID_STUB}")
            message(STATUS "Vulkan link target (Android stub): ${Vulkan_LIBRARY}")
        else()
            message(WARNING
                "No NDK libvulkan.so stub found for ${ANDROID_ABI}; linking "
                "${Vulkan_LIBRARY}, whose SONAME may not be libvulkan.so. "
                "Verify the NEEDED entry of the produced .so before packaging.")
        endif()
    endif()
    set(MNN_VULKAN_LIBS ${Vulkan_LIBRARIES} PARENT_SCOPE)
else()
    add_definitions(-DMNN_USE_LIB_WRAPPER)
endif()
IF(MNN_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    endif()
    add_library(
        MNN_Vulkan
        SHARED
        ${MNN_Vulkan_SRC}
        )
    target_link_libraries(MNN_Vulkan MNN)
    # Prevent --gc-sections from stripping the VulkanRuntimeCreator static initializer (gResistor)
    # Also ensure the .init_array from VulkanRuntime.cpp.o is included
    # Override the global -s flag to preserve .init_array entries
    target_link_options(MNN_Vulkan PRIVATE -Wl,--no-gc-sections -Wl,-u,_ZN3MNNL9gResistorE -Wl,-u,_ZN3MNNL36ensure_vulkan_runtime_registered_ptrE)
    # Override link flags to remove -s
    set_target_properties(MNN_Vulkan PROPERTIES LINK_FLAGS "")
    if (MNN_USE_SYSTEM_LIB)
        target_link_libraries(MNN_Vulkan ${Vulkan_LIBRARY})
    else()
        target_link_libraries(MNN_Vulkan ${CMAKE_DL_LIBS})
    endif()
ELSE()
    add_library(
        MNN_Vulkan
        OBJECT
        ${MNN_Vulkan_SRC}
        )
ENDIF()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include/" "${CMAKE_CURRENT_LIST_DIR}/component/" "${CMAKE_CURRENT_LIST_DIR}/runtime" "${CMAKE_CURRENT_LIST_DIR}/schema/current")
if (MNN_VULKAN_IMAGE)
    include_directories("${CMAKE_CURRENT_LIST_DIR}/image/execution" "${CMAKE_CURRENT_LIST_DIR}/image/backend" "${CMAKE_CURRENT_LIST_DIR}/image" "${CMAKE_CURRENT_LIST_DIR}/image/shaders")
else()
    include_directories("${CMAKE_CURRENT_LIST_DIR}/buffer/execution" "${CMAKE_CURRENT_LIST_DIR}/buffer/backend" "${CMAKE_CURRENT_LIST_DIR}/buffer" "${CMAKE_CURRENT_LIST_DIR}/buffer/shaders")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "^Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
endif()