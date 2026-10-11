## Goal
Apply the patch from commit `188279fe96404c5fa782dcd7a30fcde2c701ed3f` in the MNN repository, then resolve subsequent Vulkan ISP backend compilation errors — specifically generating SPIR-V bytecode and C header files for four "minimal opset" compute shaders. Later, fix GLSL 450 compatibility for all ISP shaders, remove redundant _min variants, then restore the distinction between normal and _min shaders where normal shaders use big tensor inputs and _min shaders generate the grid procedurally. Additionally, fix QNN build failure and implement fused ISP shaders (demosaic_edge+ccm, demosaic_g2+ccm).

## Constraints
(none)

## Progress
- CPU resize cache patch applied and pushed (6-file patch).
- Fixed IspSpvLookup.hpp (added MNN:: prefix to SpvData).
- Created/rewrote four minimal shader .comp source files: isp_lsc_min.comp, isp_vignetting_min.comp, isp_chromatic_aberration_min.comp, isp_raw_blc_min.comp.
- Fixed GLSL compilation issues (removed #version 450, replaced clamp() with conditional assignments, fixed sign() usage).
- Compiled all four .spv files successfully using `glslangValidator -V`.
- Generated SPIR-V header files (*_spv.h) for the minimal shaders.
- Fixed QNN build failure by adding MNN_QNN_DUMP_INTERMEDIATE_OUTPUTS flag to MNNForwardType.h.
- Fixed ISP shader rename: isp.debayer_g2 → isp.demosaic_g2 (updated IspSpvLookup.hpp and .cpp).
- Fixed uniform mismatch for isp.vignetting: updated isp_vignetting_min.comp to match non-minimal layout (8 floats).
- Fixed sign() usage in isp_chromatic_aberration_min.comp (replaced with ternary expressions).
- Fixed clamp() usage in isp_lsc.comp and isp_lsc_min.comp (replaced with conditional expressions).
- Generated headers for display shaders and other ISP shaders to ensure GLSL 450 compatibility.
- Built in build_isp and verified no compilation errors.
- Initially removed _min variant files, then restored them with proper architecture:
  * Normal opsets (isp.lsc, isp.vignetting, etc.) accept big grid/tensor input.
  * _min variants (isp.lsc_min, etc.) generate grid procedurally from parameters.
  * Both use GLSL 450 compatible code.
  * Headers placed in global namespace for proper symbol resolution.
- Implemented fused shaders:
  * isp.demosaic_edge_ccm (combines demosaic_edge and ccm).
  * isp.demosaic_g2_ccm (combines demosaic_g2 and ccm, includes display tail quantization and packing to uint8).
  * Compiled .spv files and generated headers.
  * Updated IspSpvLookup.hpp and .cpp to include the new fused shaders in the lookup map.

## Decisions
- Used `glslangValidator -V` (Termux-bundled) to compile .comp → .spv rather than relying on build-system integration.
- Rewrote shader sources with GLSL 450-compatible constructs (no clamp(), no sign() without extension).
- For _min variants, decided to keep them as procedural versions that generate the grid from parameters, while normal versions accept big tensor input.
- Fixed the namespace issue for SpvData by placing extern declarations in global namespace and using them in the map without MNN:: prefix.
- For fused shaders, created new .comp files, compiled them, generated headers, and added them to the lookup map with appropriate uniforms and bindings.

## Next steps
- Run comprehensive tests to ensure ISP shaders work correctly in all scenarios.
- Verify no performance regressions introduced by the shader changes.
- Consider cleaning up any remaining unused shader variants or temporary files.
- Monitor the build system for any further issues and address them promptly.

## Files
- Modified: source/backend/vulkan/buffer/execution/IspSpvLookup.hpp
- Modified: source/backend/vulkan/buffer/execution/IspSpvLookup.cpp
- Modified: source/core/MNNForwardType.h
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_lsc_min.comp
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_lsc_min.spv
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_lsc_min_spv.h
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_vignetting_min.comp
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_vignetting_min.spv
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_vignetting_min_spv.h
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_chromatic_aberration_min.comp
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_chromatic_aberration_min.spv
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_chromatic_aberration_min_spv.h
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_raw_blc_min.comp
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_raw_blc_min.spv
- Created/Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_raw_blc_min_spv.h
- Created: source/backend/vulkan/buffer/execution/isp_spv/isp_demosaic_edge_ccm.comp
- Created: source/backend/vulkan/buffer/execution/isp_spv/isp_demosaic_edge_ccm.spv
- Created: source/backend/vulkan/buffer/execution/isp_spv/isp_demosaic_edge_ccm_spv.h
- Created: source/backend/vulkan/buffer/execution/isp_spv/isp_demosaic_g2_ccm.comp
- Created: source/backend/vulkan/buffer/execution/isp_spv/isp_demosaic_g2_ccm.spv
- Created: source/backend/vulkan/buffer/execution/isp_spv/isp_demosaic_g2_ccm_spv.h
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_lsc.comp (clamp → conditionals)
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_lsc.spv
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_lsc_spv.h
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_vignetting.comp (uniform layout fix)
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_vignetting.spv
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_vignetting_spv.h
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_chromatic_aberration.comp (sign → ternary)
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_chromatic_aberration.spv
- Modified: source/backend/vulkan/buffer/execution/isp_spv/isp_chromatic_aberration_spv.h
