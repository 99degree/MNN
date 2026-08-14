// IspSpvLookup.hpp — Embedded SPIR-V shader lookup for isp.* ops.
// Maps isp.* op type strings (e.g., "isp.dpc") to embedded SPIR-V binary data.
// Used by VulkanFuseCreator to load shaders when the Extra op's "spirv"
// attribute is not explicitly attached by the converter.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

// Extern declarations for embedded SPIR-V data (defined in IspSpvLookup.cpp)
extern const unsigned char g_isp_dpc_spv[];
extern const int g_isp_dpc_spv_len;
extern const unsigned char g_isp_gamma_spv[];
extern const int g_isp_gamma_spv_len;
extern const unsigned char g_isp_ae_spv[];
extern const int g_isp_ae_spv_len;
extern const unsigned char g_isp_af_focus_spv[];
extern const int g_isp_af_focus_spv_len;
extern const unsigned char g_isp_awb_spv[];
extern const int g_isp_awb_spv_len;
extern const unsigned char g_isp_calib_stats_spv[];
extern const int g_isp_calib_stats_spv_len;
extern const unsigned char g_isp_denoise_spv[];
extern const int g_isp_denoise_spv_len;
extern const unsigned char g_isp_ispc_stats_spv[];
extern const int g_isp_ispc_stats_spv_len;
extern const unsigned char g_isp_tone_spv[];
extern const int g_isp_tone_spv_len;
extern const unsigned char g_isp_eis_gyro_spv[];
extern const int g_isp_eis_gyro_spv_len;
extern const unsigned char g_isp_normalize_spv[];
extern const int g_isp_normalize_spv_len;
extern const unsigned char g_isp_blc_spv[];
extern const int g_isp_blc_spv_len;
extern const unsigned char g_isp_cfa_spv[];
extern const int g_isp_cfa_spv_len;
extern const unsigned char g_isp_bayer_wb_spv[];
extern const int g_isp_bayer_wb_spv_len;
extern const unsigned char g_isp_demosaic_edge_spv[];
extern const int g_isp_demosaic_edge_spv_len;
extern const unsigned char g_isp_ccm_spv[];
extern const int g_isp_ccm_spv_len;
extern const unsigned char g_isp_downscale_spv[];
extern const int g_isp_downscale_spv_len;
extern const unsigned char g_isp_vignetting_spv[];
extern const int g_isp_vignetting_spv_len;
extern const unsigned char g_isp_saturation_spv[];
extern const int g_isp_saturation_spv_len;
extern const unsigned char g_isp_local_contrast_spv[];
extern const int g_isp_local_contrast_spv_len;
extern const unsigned char g_isp_unsharp_spv[];
extern const int g_isp_unsharp_spv_len;
extern const unsigned char g_isp_display_spv[];
extern const int g_isp_display_spv_len;
extern const unsigned char g_isp_algo_gamma_spv[];
extern const int g_isp_algo_gamma_spv_len;
extern const unsigned char g_isp_histogram_spv[];
extern const int g_isp_histogram_spv_len;

namespace MNN {

struct SpvData {
    const uint8_t* data;
    int size;
};

static const std::unordered_map<std::string, SpvData> g_ispSpvMap = {
    {"isp.dpc",         {g_isp_dpc_spv,          g_isp_dpc_spv_len}},
    {"isp.gamma",       {g_isp_gamma_spv,        g_isp_gamma_spv_len}},
    {"isp.ae",          {g_isp_ae_spv,           g_isp_ae_spv_len}},
    {"isp.af_focus",    {g_isp_af_focus_spv,     g_isp_af_focus_spv_len}},
    {"isp.awb",         {g_isp_awb_spv,          g_isp_awb_spv_len}},
    {"isp.calib_stats", {g_isp_calib_stats_spv,  g_isp_calib_stats_spv_len}},
    {"isp.denoise",     {g_isp_denoise_spv,      g_isp_denoise_spv_len}},
    {"isp.ispc_stats",  {g_isp_ispc_stats_spv,   g_isp_ispc_stats_spv_len}},
    {"isp.tone",        {g_isp_tone_spv,         g_isp_tone_spv_len}},
    {"isp.eis_gyro",    {g_isp_eis_gyro_spv,     g_isp_eis_gyro_spv_len}},
    {"isp.normalize",   {g_isp_normalize_spv,    g_isp_normalize_spv_len}},
    {"isp.blc",         {g_isp_blc_spv,          g_isp_blc_spv_len}},
    {"isp.cfa",         {g_isp_cfa_spv,          g_isp_cfa_spv_len}},
    {"isp.bayer_wb",    {g_isp_bayer_wb_spv,     g_isp_bayer_wb_spv_len}},
    {"isp.demosaic_edge",{g_isp_demosaic_edge_spv, g_isp_demosaic_edge_spv_len}},
    {"isp.ccm",         {g_isp_ccm_spv,          g_isp_ccm_spv_len}},
    {"isp.downscale",   {g_isp_downscale_spv,    g_isp_downscale_spv_len}},
    {"isp.vignetting",  {g_isp_vignetting_spv,   g_isp_vignetting_spv_len}},
    {"isp.saturation",  {g_isp_saturation_spv,   g_isp_saturation_spv_len}},
    {"isp.local_contrast",{g_isp_local_contrast_spv, g_isp_local_contrast_spv_len}},
    {"isp.unsharp",     {g_isp_unsharp_spv,      g_isp_unsharp_spv_len}},
    {"isp.display",     {g_isp_display_spv,      g_isp_display_spv_len}},
    {"isp.algo_gamma",  {g_isp_algo_gamma_spv,   g_isp_algo_gamma_spv_len}},
    {"isp.histogram",   {g_isp_histogram_spv,    g_isp_histogram_spv_len}},
};

static inline SpvData lookupIspSpv(const std::string& opType) {
    auto it = g_ispSpvMap.find(opType);
    if (it != g_ispSpvMap.end()) {
        return it->second;
    }
    return {nullptr, 0};
}

} // namespace MNN
