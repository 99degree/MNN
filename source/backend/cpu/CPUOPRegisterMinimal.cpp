// Minimal CPU op registration — only ops compiled in MNN_MINIMAL_CPU mode
#include "core/Backend.hpp"
namespace MNN {

extern void ___CPUScaleCreator__OpType_Scale__();
extern void ___CPUSoftmaxCreator__OpType_Softmax__();
extern void ___CPUCastCreator__OpType_Cast__();
extern void ___CPUMatMulCreator__OpType_MatMul__();
extern void ___CPUWhereCreator__OpType_Where__();
extern void ___CPUBinaryCreator__OpType_BinaryOp__();
extern void ___CPUUnaryCreator__OpType_UnaryOp__();
extern void ___CPUReluCreator__OpType_ReLU__();
extern void ___CPUReluCreator__OpType_PReLU__();
extern void ___CPURelu6Creator__OpType_ReLU6__();
extern void ___CPURasterFactory__OpType_Raster__();
extern void ___CPURasterFactory__OpType_While__();
extern void ___CPUEltwiseCreator__OpType_Eltwise__();
extern void ___CPUExternalConstCreator__OpType_Const__();
extern void ___CPUExternalConstCreator__OpType_TrainableParam__();

void registerCPUOps() {
    ___CPUScaleCreator__OpType_Scale__();
    ___CPUSoftmaxCreator__OpType_Softmax__();
    ___CPUCastCreator__OpType_Cast__();
    ___CPUMatMulCreator__OpType_MatMul__();
    ___CPUWhereCreator__OpType_Where__();
    ___CPUBinaryCreator__OpType_BinaryOp__();
    ___CPUUnaryCreator__OpType_UnaryOp__();
    ___CPUReluCreator__OpType_ReLU__();
    ___CPUReluCreator__OpType_PReLU__();
    ___CPURelu6Creator__OpType_ReLU6__();
    ___CPURasterFactory__OpType_Raster__();
    ___CPURasterFactory__OpType_While__();
    ___CPUEltwiseCreator__OpType_Eltwise__();
    ___CPUExternalConstCreator__OpType_Const__();
    ___CPUExternalConstCreator__OpType_TrainableParam__();
}
}
