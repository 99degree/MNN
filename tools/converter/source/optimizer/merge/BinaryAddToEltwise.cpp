//
//  BinaryAddToEltwise.cpp
//  MNNConverter
//
//  Created by MNN on 2020/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

// Registration function to avoid static initialization order issues
static void registerBinaryAddToEltwise() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_BinaryOp) {
            return false;
        }
        auto binaryOp     = expr->get();
        auto binaryParams = binaryOp->main_as_BinaryOp();
        if (binaryParams->opType() != BinaryOpOperation_ADD) {
            return false;
        }
        
        auto leftConvertVar  = expr->inputs()[0];
        auto leftConvertExpr = leftConvertVar->expr().first;
        if (leftConvertExpr->get() == nullptr) {
            return false;
        }
        if (leftConvertExpr->get()->type() != OpType_ConvertTensor) {
            return false;
        }
        
        auto leftMoreConvertExpr = leftConvertExpr->inputs()[0]->expr().first;
        while (leftMoreConvertExpr->get() && leftMoreConvertExpr->get()->type() == OpType_ConvertTensor) {
            leftConvertExpr     = leftMoreConvertExpr;
            leftMoreConvertExpr = leftConvertExpr->inputs()[0]->expr().first;
        }
        
        auto leftInt8ToFloatVar  = leftConvertExpr->inputs()[0];
        auto leftInt8ToFloatExpr = leftInt8ToFloatVar->expr().first;
        if (leftInt8ToFloatExpr->get() == nullptr) {
            return false;
        }
        if (leftInt8ToFloatExpr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }
        
        auto rightConvertVar  = expr->inputs()[1];
        auto rightConvertExpr = rightConvertVar->expr().first;
        if (rightConvertExpr->get() == nullptr) {
            return false;
        }
        if (rightConvertExpr->get()->type() != OpType_ConvertTensor) {
            return false;
        }
        
        auto rightMoreConvertExpr = rightConvertExpr->inputs()[0]->expr().first;
        while (rightMoreConvertExpr->get() && rightMoreConvertExpr->get()->type() == OpType_ConvertTensor) {
            rightConvertExpr     = rightMoreConvertExpr;
            rightMoreConvertExpr = rightConvertExpr->inputs()[0]->expr().first;
        }
        
        auto rightInt8ToFloatVar  = rightConvertExpr->inputs()[0];
        auto rightInt8ToFloatExpr = rightInt8ToFloatVar->expr().first;
        if (rightInt8ToFloatExpr->get() == nullptr) {
            return false;
        }
        if (rightInt8ToFloatExpr->get()->type() != OpType_Int8ToFloat) {
            return false;
        }
        
        return true;
    };
    
    auto transform = [](EXPRP expr) {
        auto leftConvertVar  = expr->inputs()[0];
        auto leftConvertExpr = leftConvertVar->expr().first;
        auto leftConvertOp   = leftConvertExpr->get();
        
        auto leftInt8ToFloatVar  = leftConvertExpr->inputs()[0];
        auto leftInt8ToFloatExpr = leftInt8ToFloatVar->expr().first;
        
        auto rightConvertVar  = expr->inputs()[1];
        auto rightConvertExpr = rightConvertVar->expr().first;
        auto rightConvertOp   = rightConvertExpr->get();
        
        auto rightInt8ToFloatVar  = rightConvertExpr->inputs()[0];
        auto rightInt8ToFloatExpr = rightInt8ToFloatVar->expr().first;
        
        auto eltwiseSum = _Sum(leftInt8ToFloatVar, rightInt8ToFloatVar, {});
        eltwiseSum->setName(expr->name());
        
        Expr::replace(expr, eltwiseSum->expr().first);
        return true;
    };
    
    TemplateMerge::getInstance("Merge").insertTemplate("BinaryAddToEltwise", match, transform, PASS_PRIORITY_MIDDLE);
}


// Explicit initialization function to be called by PostConverter
MNN_PUBLIC void __attribute__((visibility("default"))) ensureBinaryAddToEltwiseRegistered() {
    // Intentionally a no-op: the BinaryAddToEltwise merge pass is optional and
    // its TemplateMerge registration crashed during static init on Android
    // (fault in std::map tree search while inserting into gMerge). Our ISP
    // pass0 pipeline never emits Int8ToFloat+Convert+ADD patterns, so the pass
    // is never needed. The pass body (registerBinaryAddToEltwise) is retained
    // for future debugging; the symbol is exported so PostConverter's call
    // and the linker -u option still resolve.
    (void)0;
}

} // namespace Express
} // namespace MNN
