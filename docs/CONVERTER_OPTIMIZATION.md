# MNN Converter: ONNX→MNN vs MNN→MNN Optimization Behavior

## 1. Convert Entry API

The converter is exposed as a C++ class, not a standalone `extern "C"` symbol:

```cpp
// Parse argv / populate modelConfig
MNN::Cli::initializeMNNConvertArgs(modelConfig& modelPath, int argc, char** argv);

// Run the conversion
MNN::Cli::convertModel(modelConfig& modelPath);
```

PyMNN wraps these through `pymnn/src/MNNTools.cc`.

## 2. `optimizeLevel` Semantics

`modelConfig::optimizeLevel` (default `1`) is parsed from the `--optimizeLevel` CLI flag.

### 2.1 ONNX → MNN (and TF / Caffe / TFLite / Torch → MNN)

```cpp
// cli.cpp:691
bool needOptimize = modelPath.model != modelConfig::MNN || modelPath.optimizeLevel >= 1;
```

For **any non-MNN source**, `needOptimize` is unconditionally `true`.  
The full `optimizeNet()` pipeline runs regardless of `optimizeLevel`.

```cpp
// cli.cpp:697
if (1 == modelPath.optimizeLevel && modelPath.model == modelConfig::MNN) {
    expectedPass = { "TranslateJsonOp", "FuseDupOp", "RemoveInvalidCast", "RemoveDeadShapeOp" };
}
```

`expectedPass` is **only** populated for `MNN → MNN`. For ONNX→MNN it is empty, so `optimizeNet()` runs the full default pass set.

### 2.2 MNN → MNN

| optimizeLevel | Behavior |
|---|---|
| `0` | Skips optimization entirely (`needOptimize = false`). |
| `1` | Runs a minimal 4-pass subset only. No ISP fusions. |
| `> 1` | Runs the full optimization pipeline, including ISP fusions if Extra ops exist. |

## 3. ISP Fusion Activation

ISP fusions (`IspChainFusion` pass) are **not** gated by `optimizeLevel`.  
They activate when the model contains Extra ops (e.g., ONNX `isp.*` ops converted by `IspOnnxOps` into `OpType_Extra`).

Control flags (all part of `modelConfig`):

| Flag | Effect |
|---|---|
| `--isp_fusion=<enable\|disable\|N>` (`ispFusionMeta`) | Enable/disable the pass, set pattern threshold. |
| `--isp_fusion_threshold=N` (`ispFusionThreshold`) | Max pattern id to try during dispatch. |
| `--dumpPass` (`dumpPass`) | Per-pass verbose diagnostics. |

## 4. All `modelConfig` Flags That Affect Optimization

| Flag | Default | Effect |
|---|---|---|
| `optimizeLevel` | `1` | Gates optimization for `MNN → MNN` only. |
| `saveStaticModel` | `false` | Skips optimization for `MNN → MNN`. |
| `forTraining` | `false` | Selects training-safe optimization paths. |
| `ispFusionMeta` | `""` | ISP fusion metadata from ONNX `metadata_props`. |
| `ispFusionThreshold` | `999` | Max pattern id for ISP dispatch. |
| `dumpPass` | `false` | Verbose per-pass output. |
| `convertMatmulToConv` | `false` | Convert MatMul to Conv where profitable. |
| `groupConvNative` | `false` | Prefer native group convolution. |
| `detectSparseSpeedUp` | `false` | Enable sparse speed-up detection. |
| `alignDenormalizedValue` | `false` | Align denormalized float values. |
| `keepInputFormat` | `false` | Preserve original input tensor format. |
| `optimizePrefer` | `false` | Prefer speed over size. |
| `splitQuantBlock` | `false` | Split quantized convolution blocks. |
| `transformerFuseC4` | `true` | Fuse Transformer C4 pattern (MNN→MNN only). |
| `defaultBatchSize` | `0` | Override dynamic batch dim. |

## 5. Current Limitation: No ONNX→MNN Optimization Level

There is **no flag** to downgrade ONNX→MNN to a reduced pass list.  
`optimizeLevel` is effectively ignored for non-MNN inputs — the full pipeline always runs.

### Where to change it

If a tunable optimization level for ONNX→MNN is desired, the relevant code paths are:

- **Gate `needOptimize`**: `tools/converter/source/common/cli.cpp:691`
- **Populate `expectedPass` for non-MNN**: `tools/converter/source/common/cli.cpp:697`
- **Pass selection logic**: `tools/converter/source/optimizer/PostConverter.cpp:648` (`optimizeNet()`)

A typical change would branch on `modelPath.optimizeLevel` for all source types, mapping levels to explicit pass lists (e.g., level 0 = skip, level 1 = core fusions only, level 2 = full pipeline).
