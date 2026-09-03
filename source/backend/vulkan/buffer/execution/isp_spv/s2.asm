; SPIR-V
; Version: 1.0
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 116
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 16 16 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %Uniforms BufferBlock
               OpMemberDecorate %Uniforms 0 Offset 0
               OpMemberDecorate %Uniforms 1 Offset 4
               OpMemberDecorate %Uniforms 2 Offset 8
               OpMemberDecorate %Uniforms 3 Offset 12
               OpDecorate %u Binding 0
               OpDecorate %u DescriptorSet 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpDecorate %OutBuf BufferBlock
               OpMemberDecorate %OutBuf 0 NonReadable
               OpMemberDecorate %OutBuf 0 Offset 0
               OpDecorate %_ NonReadable
               OpDecorate %_ Binding 2
               OpDecorate %_ DescriptorSet 0
               OpDecorate %_runtimearr_float_0 ArrayStride 4
               OpDecorate %InBuf BufferBlock
               OpMemberDecorate %InBuf 0 NonWritable
               OpMemberDecorate %InBuf 0 Offset 0
               OpDecorate %__0 NonWritable
               OpDecorate %__0 Binding 1
               OpDecorate %__0 DescriptorSet 0
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
%_ptr_Input_uint = OpTypePointer Input %uint
     %uint_1 = OpConstant %uint 1
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %float = OpTypeFloat 32
   %Uniforms = OpTypeStruct %float %float %float %float
%_ptr_Uniform_Uniforms = OpTypePointer Uniform %Uniforms
          %u = OpVariable %_ptr_Uniform_Uniforms Uniform
      %int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
      %int_1 = OpConstant %int 1
       %bool = OpTypeBool
      %int_4 = OpConstant %int 4
%_runtimearr_float = OpTypeRuntimeArray %float
     %OutBuf = OpTypeStruct %_runtimearr_float
%_ptr_Uniform_OutBuf = OpTypePointer Uniform %OutBuf
          %_ = OpVariable %_ptr_Uniform_OutBuf Uniform
%_runtimearr_float_0 = OpTypeRuntimeArray %float
      %InBuf = OpTypeStruct %_runtimearr_float_0
%_ptr_Uniform_InBuf = OpTypePointer Uniform %InBuf
        %__0 = OpVariable %_ptr_Uniform_InBuf Uniform
      %int_2 = OpConstant %int 2
      %int_3 = OpConstant %int 3
    %uint_16 = OpConstant %uint 16
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_16 %uint_16 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %x = OpVariable %_ptr_Function_uint Function
          %y = OpVariable %_ptr_Function_uint Function
          %w = OpVariable %_ptr_Function_int Function
          %h = OpVariable %_ptr_Function_int Function
        %idx = OpVariable %_ptr_Function_int Function
       %base = OpVariable %_ptr_Function_int Function
         %14 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
         %15 = OpLoad %uint %14
               OpStore %x %15
         %18 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_1
         %19 = OpLoad %uint %18
               OpStore %y %19
         %29 = OpAccessChain %_ptr_Uniform_float %u %int_0
         %30 = OpLoad %float %29
         %31 = OpConvertFToS %int %30
               OpStore %w %31
         %34 = OpAccessChain %_ptr_Uniform_float %u %int_1
         %35 = OpLoad %float %34
         %36 = OpConvertFToS %int %35
               OpStore %h %36
         %38 = OpLoad %uint %x
         %39 = OpBitcast %int %38
         %40 = OpLoad %int %w
         %41 = OpSGreaterThanEqual %bool %39 %40
         %42 = OpLogicalNot %bool %41
               OpSelectionMerge %44 None
               OpBranchConditional %42 %43 %44
         %43 = OpLabel
         %45 = OpLoad %uint %y
         %46 = OpBitcast %int %45
         %47 = OpLoad %int %h
         %48 = OpSGreaterThanEqual %bool %46 %47
               OpBranch %44
         %44 = OpLabel
         %49 = OpPhi %bool %41 %5 %48 %43
               OpSelectionMerge %51 None
               OpBranchConditional %49 %50 %51
         %50 = OpLabel
               OpReturn
         %51 = OpLabel
         %54 = OpLoad %uint %y
         %55 = OpBitcast %int %54
         %56 = OpLoad %int %w
         %57 = OpIMul %int %55 %56
         %58 = OpLoad %uint %x
         %59 = OpBitcast %int %58
         %60 = OpIAdd %int %57 %59
               OpStore %idx %60
         %62 = OpLoad %int %idx
         %64 = OpIMul %int %62 %int_4
               OpStore %base %64
         %69 = OpLoad %int %idx
         %74 = OpLoad %int %base
         %75 = OpIAdd %int %74 %int_0
         %76 = OpAccessChain %_ptr_Uniform_float %__0 %int_0 %75
         %77 = OpLoad %float %76
         %78 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %69
               OpStore %78 %77
         %79 = OpLoad %int %w
         %80 = OpLoad %int %h
         %81 = OpIMul %int %79 %80
         %82 = OpIMul %int %81 %int_1
         %83 = OpLoad %int %idx
         %84 = OpIAdd %int %82 %83
         %85 = OpLoad %int %base
         %86 = OpIAdd %int %85 %int_1
         %87 = OpAccessChain %_ptr_Uniform_float %__0 %int_0 %86
         %88 = OpLoad %float %87
         %89 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %84
               OpStore %89 %88
         %90 = OpLoad %int %w
         %91 = OpLoad %int %h
         %92 = OpIMul %int %90 %91
         %94 = OpIMul %int %92 %int_2
         %95 = OpLoad %int %idx
         %96 = OpIAdd %int %94 %95
         %97 = OpLoad %int %base
         %98 = OpIAdd %int %97 %int_2
         %99 = OpAccessChain %_ptr_Uniform_float %__0 %int_0 %98
        %100 = OpLoad %float %99
        %101 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %96
               OpStore %101 %100
        %102 = OpLoad %int %w
        %103 = OpLoad %int %h
        %104 = OpIMul %int %102 %103
        %106 = OpIMul %int %104 %int_3
        %107 = OpLoad %int %idx
        %108 = OpIAdd %int %106 %107
        %109 = OpLoad %int %base
        %110 = OpIAdd %int %109 %int_3
        %111 = OpAccessChain %_ptr_Uniform_float %__0 %int_0 %110
        %112 = OpLoad %float %111
        %113 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %108
               OpStore %113 %112
               OpReturn
               OpFunctionEnd
