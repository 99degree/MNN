// IspOpsetValidate.hpp — Canonical isp.* opset validity table + trap helpers.
//
// PURPOSE
//   `isp.*` Extra ops are only executable when the name matches a runtime
//   SPIR-V shader key in source/backend/vulkan/buffer/execution/IspSpvLookup.hpp.
//   A typo / stale literal (e.g. "isp.algo_gamma" vs "isp.algogamma") produces an
//   Extra op WITHOUT a shader -> VulkanFuseCreator returns nullptr -> the op
//   silently falls back to CPU (or fails), which is very hard to notice.
//
//   This header is the single source of truth used to TRAP that condition at
//   two points:
//     - CONVERT time : tools/converter/.../onnxOpConverter.cpp
//     - SESSION time : source/core/Interpreter.cpp
//
// SYNC
//   kIspOpsetNames MUST stay in sync with the keys of g_ispSpvMap in
//   source/backend/vulkan/buffer/execution/IspSpvLookup.hpp. When a shader is
//   added/renamed there, update this list.
//
//   Escalation: set MNN_ISP_TRAP=1 to make an invalid name a hard error
//   (session creation fails) instead of a loud log line.
//
//   Two failure modes are covered:
//     - INVALID : an isp.* name with no matching shader key (typo / drift)
//     - MISSING : a model that carries ZERO isp.* ops, i.e. every isp block
//                 silently degraded to CPU primitives so the whole ISP chain
//                 never reaches VulkanFuse
#pragma once

#include <cstdlib>
#include <string>

namespace MNN {

// Canonical isp.* op names (with the "isp." prefix), matching IspSpvLookup keys.
inline const char* const kIspOpsetNames[] = {
    "isp.aaf",
    "isp.ae",
    "isp.af_focus",
    "isp.algo_gamma",
    "isp.algo_gamma_scalar",
    "isp.algo_ae",
    "isp.algo_cct",
    "isp.algo_awb",
    "isp.auto_contrast",
    "isp.awb",
    "isp.bad_pixel_lsc_bayer_wb",
    "isp.bayer_wb",
    "isp.bilateral",
    "isp.blc",
    "isp.bnf",
    "isp.calib_stats",
    "isp.ccm",
    "isp.cct",
    "isp.ceh",
    "isp.cfa",
    "isp.chromatic_aberration",
    "isp.chromatic_aberration_min",
    "isp.cnf",
    "isp.colorspace",
    "isp.crop",
    "isp.crop_blc",
    "isp.crop_blc_norm_cfa",
    "isp.demosaic_ccm",
    "isp.demosaic_edge",
    "isp.demosaic_edge_ccm",
    "isp.demosaic_g2",
    "isp.demosaic_g2_ccm",
    "isp.denoise",
    "isp.display",
    "isp.display_rgb",
    "isp.display_uint32_argb",
    "isp.display_uint32_bgra",
    "isp.display_uint32_rgba",
    "isp.display_uint8_argb",
    "isp.display_uint8_bgra",
    "isp.display_uint8_rgba",
    "isp.downscale",
    "isp.dpc",
    "isp.ee",
    "isp.eis_gyro",
    "isp.fcs",
    "isp.gamma",
    "isp.gpu_warp",
    "isp.hdr_merge",
    "isp.histogram",
    "isp.ispc_stats",
    "isp.ldci",
    "isp.local_contrast",
    "isp.lsc",
    "isp.lsc_min",
    "isp.nlm",
    "isp.normalize",
    "isp.pipeline",
    "isp.pyramid",
    "isp.raw_blc",
    "isp.raw_blc_min",
    "isp.blc_min",
    "isp.demosaic_malvar",
    "isp.saturation",
    "isp.stats",
    "isp.temporal_denoise",
    "isp.tone",
    "isp.tone_stats",
    "isp.unsharp",
    "isp.vignetting",
    "isp.vignetting_min",
    "isp.warp",
    "isp.yuv_bcc",
    "isp.yuv_sat",
    "isp.zone_stats",
};

// True when `type` is an isp.* op name present in the canonical table.
inline bool isKnownIspOp(const std::string& type) {
    for (const char* name : kIspOpsetNames) {
        if (type == name) {
            return true;
        }
    }
    return false;
}

// True when the caller asked for a hard failure on invalid isp.* names.
inline bool ispOpsetTrapEnabled() {
    const char* v = std::getenv("MNN_ISP_TRAP");
    return v != nullptr && v[0] == '1';
}

// True when the caller declares that the model under inspection is an ISP model
// and therefore MUST contain at least one isp.* op. MNN_ISP_EXPECT=1 arms the
// MISSING check (zero isp.* ops => trap); it can also carry an explicit minimum
// count (e.g. MNN_ISP_EXPECT=6). Unset => MISSING check is informational only.
inline bool ispExpectEnabled(int* minCount) {
    const char* v = std::getenv("MNN_ISP_EXPECT");
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    int n = std::atoi(v);
    if (minCount != nullptr) {
        *minCount = (n > 0) ? n : 1;
    }
    return true;
}

// Set by the ONNX importer when the source graph contained at least one node in
// the custom `isp` domain (or an `isp.*` op_type). Such a graph is by definition
// an ISP model, so the converted MNN model MUST still carry isp.* Extra ops.
// This lets the converter raise a MISSING trap even though the ONNX graph itself
// no longer mentions the (possibly dropped) concrete op names.
inline bool& ispModelFlag() {
    static bool sFlag = false;
    return sFlag;
}

} // namespace MNN
