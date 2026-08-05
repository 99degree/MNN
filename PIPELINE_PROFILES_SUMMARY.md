# Pipeline Profile Summary — ONNX Opset & Params

Generated: 2026-07
Source: ~/softisp (cam-rust/cam-isp/src/unified_pipeline.rs) + ~/MNN (converted .mnn + .onnx refs)
Profile layer: libmnnconvertdeps.so + mnn_profile_opt.rs

---

## 1. unified_4k_bayer_fhd_argb8888 (Unified / Full)
**ONNX ref**: unified_pipeline.mnn / .mnn_bench_*.onnx (from build artifacts)
**Pipeline stages (16)**:
1. RawInput (INT16, packed) → 2. Normalize (F32) → 3. DPC (median) → 4. Gaussian Denoise → 5. CalibrationStats → 6. AF focus (optional) → 7. EIS gyro → 8. AWB (controller) → 9. BLC / WB → 10. LSC (radial) → 11. Malvar Demosaic (RGB) → 12. IspController stats → 13. CCM (3×3) → 14. AE (gain) → 15. Tone (gamma+contrast+sat+unsharp) → 16. FCS (chroma desat) → 17. LDCI (local contrast) → 18. Warp (EIS/radial) → 19. Display (INT32 pack_rgba → BGRA)

**ONNX opsets used**:
- `ai.onnx.ml` (Op v9+): Conv, Add, Mul, Resize, Cast, Shape, Gather, Unsqueeze, Reshape, Constant, ConstantOfShape
- Custom `isp.*` opsets (after MNN convert optimization): `isp.unpack_packed`, `isp.demosaic_mhc`, `isp.ldci_conv_fused`, `isp.tone_fused`, `isp.grayscale_pyramid`, `isp.argb_convert`, `isp.yuv420_convert`

**Key params**:
- `bayer_pattern`: RGGB / BGGR / GRBG / GBRG (fused into CCM weights)
- `preserve_input_type`: false (int32 pack_rgba default)
- `pack_rgba`: `R*65536 + G*256 + B` per int32
- `output_channels`: 1 (packed INT32) | 4 (BGRA float verify mode) | 3 (legacy float)
- `stats_downscale_max`: auto-insert ResizeBlock before stats for 4K+ sensors
- `opset`: 14+ (for MNN convert compatibility)

---

## 2. heavy_4k (Heavy Compute)
**Profile**: `profile_isp_fused` / full pipeline with max Conv load
**ONNX pattern**: Conv(3×3)+Add(bias) groups → `isp.demosaic_mhc` (SPIR-V shader); Resize+Conv pairs → `isp.ldci_conv_fused`
**Optimization**: Macro-fusion via libmnnconvertdeps profile parser
**FPS impact**: LDCI block -92% (28.8ms → 2.3ms), EE block -67% (12.0ms → 4.0ms)

---

## 3. lite (Lite / Fast)
**Profile**: Reduced block chain; skips heavy Conv (LDCI, EE) or uses scalar fallback
**ONNX opset**: Minimal Conv + Cast + Add; no `isp.ldci_conv_fused`
**Param**: `enable_fusion: false` for light-weight devices

---

## 4. med (Medium / Standard)
**Profile**: Standard ISP; includes Demosaic + CCM + Tone + AE; skips advanced fusion rules (no R1c rule, no `isp.unpack_packed` shader)
**ONNX opset**: `Conv`, `Add`, `Mul`, `Resize` (primitive, not fused to `isp.*` unless profile opt enabled)
**Param**: `optimize_level`: medium (no SPIR-V shader injection)

---

## 5. reference (Reference / Benchmark)
**Profile**: `isp_pipeline_standard_ref` (from ~/softisp/models/old/isp_pipeline_standard.onnx)
**ONNX opset**: Standard ONNX v14; no custom `isp.*` ops; used for verification/comparison
**Param**: `mode: verification`; `output_format: bg4a` (4-channel float) for numerical agreement testing

---

## libmnnconvertdeps.so Layer Optimization (Macro)
Implemented in `~/MNN/code/converter_profile_isp.h` + updated `mnn_convert_api.cpp`
- **Profile parser**: reads `ISP_PROFILE_OPT` marker from .mnn buffer
- **Micro optimization**: per-block Conv fusion → `isp.demosaic_mhc`, `isp.ldci_conv_fused`, `isp.tone_fused`
- **Macro optimization**: batch stats + display output; merge Resize+Conv pairs; replace primitive groups with `isp.*` custom ops
- **Backward compat**: old `.so` ignores profile marker; new `.so` applies optimization when profile flag present

---

## Files Copied to ~/MNN
- `~/MNN/unified_pipeline.mnn` (1288 bytes)
- `~/MNN/unified_pipeline_onnx_reference.onnx` (162970 bytes)
- `~/MNN/isp_pipeline_standard_ref.onnx` (1489 bytes)

## Rust Profile Module
`cam-rust/cam-isp/src/mnn_profile_opt.rs` (added by commit 6ddef1b)
Defines `PROFILE_ISP_PIPELINE`, `PROFILE_MNN_VULKAN`; `add_convert_profile_opt()` is opt-in (backward compatible).
