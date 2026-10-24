//
//  MNNIspTrace.hpp — shared declaration of the ISP trace sink.
//
//  WHY THIS EXISTS
//    `ispTraceSink` began as a file-local helper in VulkanFuse.cpp. Several
//    translation units declared it inline as `extern FILE* ispTraceSink();`.
//    Those block-scope declarations with C++ linkage still produced an
//    undefined GLOBAL symbol in libMNN_Vulkan.so, so dlopen of the library
//    (and of libmnnhelper.so, which links it) failed with
//      "cannot locate symbol ispTraceSink referenced by libmnnhelper.so".
//    Declaring it ONCE here as an exported C symbol gives every user the same
//    definition and no hidden cross-object assumption.
//
#ifndef MNNIspTrace_hpp
#define MNNIspTrace_hpp

#include <cstdio>

extern "C" __attribute__((visibility("default"))) FILE* mnnIspTraceSink();

#endif /* MNNIspTrace_hpp */
