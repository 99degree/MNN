# Non-Standard ONNX Format Issue Analysis

## Summary

The `pass0_crash.onnx` file captured from the Android app crash is **not a standard ONNX ModelProto**. It contains a custom header format that prevents the MNN converter from parsing it correctly.

## File Format Analysis

### File: `/sdcard/Download/pass0_crash.onnx`
- **Size**: 13,143 bytes
- **Magic bytes**: `08 07 12 13` (4 bytes)

### Actual Structure (Protobuf with Custom Wrapper)

The file is a protobuf message with this structure:

```
Field 1 (tag 0x08): ir_version = 7 (varint)
Field 2 (tag 0x12): producer_name = "cam_isp_onnx_engine" (string)
Field 7 (tag 0x3A): Nested message (119 bytes) - appears to be the actual ONNX graph data
  Nested Field 2: "raw_input_pipeline"
  Nested Field 1: Node names like "Slice", "CropBlock/starts", etc.
Field 12 (tag 0x62): START GROUP (deprecated protobuf syntax)
```

## Why MNN Converter Fails

The MNN converter expects a **standard ONNX ModelProto** with these fields:
- Field 1: `ir_version` (uint32)
- Field 2: `producer_name` (string)  
- Field 3: `model_version` (int64)
- Field 4: `graph` (GraphProto message) - **THE ACTUAL MODEL**
- Field 5: `opset_import` (repeated OpSetIdProto)

But the crash file has:
- Field 1: `ir_version = 7` ✓
- Field 2: `producer_name = "cam_isp_onnx_engine"` (custom, not standard)
- Field 7: Nested message with graph data (non-standard field number)
- **No Field 4 (graph) at standard location**

## Root Cause

The Android app's `OnnxPreprocessor.rewriteForBackend()` or the ONNX composition pipeline wraps the actual ONNX model in a custom protobuf envelope with:
1. A 4-byte magic header (`08 07 12 13`)
2. A custom `producer_name` identifying the pipeline (`cam_isp_onnx_engine`)
3. The actual ONNX graph embedded at Field 7 instead of Field 4

## Solution Options

### Option 1: Strip Custom Header (Recommended)
Extract the nested message at Field 7 and feed only that to the converter:

```python
# The actual ONNX ModelProto starts after the custom header
# Magic (4 bytes) + producer_name field (varies) = ~24 bytes
# Then the nested message at Field 7 contains the real model
```

### Option 2: Modify Converter
Add a pre-parser in `MNNConvertBuffer.cpp` that detects this format and extracts the inner ModelProto before calling `ParseFromArray`.

### Option 3: Fix Upstream
Modify the Android app's `OnnxPreprocessor` to output standard ONNX without the custom wrapper.

## Verification

Created test files:
- `pass0_crash_protobuf.onnx` - Extracted protobuf (13,119 bytes) - **Still fails** because it's the wrapper, not the inner model
- The real model is nested deeper at Field 7

## Impact

- **Converter correctly rejects** the file with "failed to parse ONNX protobuf from memory"
- This is **expected behavior** - the file is not valid ONNX
- The production crash was likely caused by a different valid ONNX that triggered a real bug
- Need to capture a **valid ONNX** from the app for proper crash reproduction

## Next Steps

1. Add debug logging in Android app to dump the ONNX **before** `OnnxPreprocessor.rewriteForBackend()` 
2. Or modify `MnnConvert.kt` to save the preprocessed ONNX bytes
3. Use that valid ONNX for reproducer testing
4. The current ISP shader fixes are complete and verified; the ONNX format issue is separate