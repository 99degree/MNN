#include "IspSpvLookup.hpp"

// Force default visibility for the embedded SPIR-V data so VulkanFuse can
// access it at runtime. The MNN Vulkan backend compiles with -fvisibility=hidden,
// which would otherwise hide these symbols.
#pragma GCC visibility push(default)
#include "isp_spv/isp_dpc_spv.h"
#include "isp_spv/isp_gamma_spv.h"
#include "isp_spv/isp_ae_spv.h"
#include "isp_spv/isp_af_focus_spv.h"
#include "isp_spv/isp_awb_spv.h"
#include "isp_spv/isp_calib_stats_spv.h"
#include "isp_spv/isp_denoise_spv.h"
#include "isp_spv/isp_ispc_stats_spv.h"
#include "isp_spv/isp_tone_spv.h"
#include "isp_spv/isp_eis_gyro_spv.h"
#include "isp_spv/isp_histogram_spv.h"
#include "isp_spv/isp_algo_gamma_spv.h"
#include "isp_spv/isp_display_spv.h"
#include "isp_spv/isp_unsharp_spv.h"
#include "isp_spv/isp_local_contrast_spv.h"
#include "isp_spv/isp_saturation_spv.h"
#include "isp_spv/isp_vignetting_spv.h"
#include "isp_spv/isp_downscale_spv.h"
#include "isp_spv/isp_ccm_spv.h"
#include "isp_spv/isp_demosaic_edge_spv.h"
#include "isp_spv/isp_bayer_wb_spv.h"
#include "isp_spv/isp_cfa_spv.h"
#include "isp_spv/isp_blc_spv.h"
#include "isp_spv/isp_normalize_spv.h"
#pragma GCC visibility pop
