//
//  IspOnnxOps.cpp — Custom isp.ai ONNX ops for ISP pipeline
//  Transforms isp.ai custom ops to VulkanFuse Extra ops with embedded SPIR-V
//
#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include <string>
#include <vector>
#include "OnnxExtraManager.hpp"
#include "MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

// Helper: create a VulkanFuse Extra op from ONNX attributes
VARP createVulkanFuseOp(EXPRP expr, const Op* onnxOp, const std::string& typeName) {
    auto extraOp = std::unique_ptr<OpT>(new OpT);
    extraOp->type = OpType_Extra;
    extraOp->main.type = OpParameter_Extra;
    extraOp->main.value = new ExtraT;

    auto* extra = static_cast<ExtraT*>(extraOp->main.value);
    extra->type = "isp." + typeName;  // e.g., "isp.unpack_blc"

    // Copy attributes from ONNX op to Extra op
    auto* onnxExtra = onnxOp->main_as_Extra();
    for (int i = 0; i < onnxExtra->attr()->size(); ++i) {
        auto srcAttr = onnxExtra->attr()->GetAs<Attribute>(i);
        std::unique_ptr<AttributeT> dstAttr(new AttributeT);
        dstAttr->key = srcAttr->key()->str();
        dstAttr->i = srcAttr->i();
        dstAttr->b = srcAttr->b();
        dstAttr->f = srcAttr->f();
        dstAttr->s = srcAttr->s()->str();

        if (auto t = srcAttr->tensor()) {
            dstAttr->tensor.reset(new BlobT);
            dstAttr->tensor->dataType = t->dataType();
            dstAttr->tensor->dataFormat = t->dataFormat();
            if (auto d = t->int32s()) {
                dstAttr->tensor->int32s.resize(d->size());
                memcpy(dstAttr->tensor->int32s.data(), d->data(), d->size() * sizeof(int32_t));
            }
            if (auto d = t->float32s()) {
                dstAttr->tensor->float32s.resize(d->size());
                memcpy(dstAttr->tensor->float32s.data(), d->data(), d->size() * sizeof(float));
            }
            if (auto d = t->int8s()) {
                dstAttr->tensor->int8s.resize(d->size());
                memcpy(dstAttr->tensor->int8s.data(), d->data(), d->size() * sizeof(int8_t));
            }
        }
        if (auto lst = srcAttr->list()) {
            dstAttr->list.reset(new ListValueT);
            if (auto d = lst->i()) {
                dstAttr->list->i.resize(d->size());
                memcpy(dstAttr->list->i.data(), d->data(), d->size() * sizeof(int32_t));
            }
            if (auto d = lst->f()) {
                dstAttr->list->f.resize(d->size());
                memcpy(dstAttr->list->f.data(), d->data(), d->size() * sizeof(float));
            }
            if (auto d = lst->s()) {
                for (int j = 0; j < d->size(); ++j) {
                    dstAttr->list->s.push_back(d->GetAsString(j)->str());
                }
            }
        }
        extra->attr.push_back(std::move(dstAttr));
    }

    // Create input variables
    std::vector<VARP> inputs;
    for (int i = 0; i < expr->inputs().size(); ++i) {
        inputs.push_back(expr->inputs()[i]);
    }

    return Variable::create(Expr::create(extraOp.get(), inputs));
}

// ── isp.ai::UnpackBlc → isp.unpack_blc Extra op ──
class IspUnpackBlc : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "unpack_blc");
        return newVar->expr().first;
    }
};

// ── isp.ai::DemosaicNoscale → isp.demosaic_noscale Extra op ──
class IspDemosaicNoscale : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "demosaic_noscale");
        return newVar->expr().first;
    }
};

// ── isp.ai::Fcs → isp.fcs Extra op ──
class IspFcs : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "fcs");
        return newVar->expr().first;
    }
};

// ── isp.ai::Ee → isp.ee Extra op ──
class IspEe : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "ee");
        return newVar->expr().first;
    }
};

// ── isp.ai::Ldci → isp.ldci Extra op ──
class IspLdci : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "ldci");
        return newVar->expr().first;
    }
};

// ── isp.ai::Display → isp.display Extra op ──
class IspDisplay : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "display");
        return newVar->expr().first;
    }
};

// ── isp.ai::FcsDisplay → isp.fcs_display fused Extra op ──
class IspFcsDisplay : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "fcs_display");
        return newVar->expr().first;
    }
};

// ── isp.ai::EeLdci → isp.ee_ldci fused Extra op ──
class IspEeLdci : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "ee_ldci");
        return newVar->expr().first;
    }
};

// ── isp.ai::Vignetting → isp.vignetting Extra op ──
class IspVignetting : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "vignetting");
        return newVar->expr().first;
    }
};

// ── isp.ai::AutoContrast → isp.auto_contrast Extra op ──
class IspAutoContrast : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "auto_contrast");
        return newVar->expr().first;
    }
};

// ── isp.ai::Colorspace → isp.colorspace Extra op ──
class IspColorspace : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "colorspace");
        return newVar->expr().first;
    }
};

// ── isp.ai::WaveletDenoise → isp.wavelet_denoise Extra op ──
class IspWaveletDenoise : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "wavelet_denoise");
        return newVar->expr().first;
    }
};

// ── isp.ai::Bilateral → isp.bilateral Extra op ──
class IspBilateral : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "bilateral");
        return newVar->expr().first;
    }
};

// ── Static registration ──
static bool gReg = []() {
    auto mgr = OnnxExtraManager::get();
    mgr->insert("UnpackBlc", std::make_shared<IspUnpackBlc>());
    mgr->insert("DemosaicNoscale", std::make_shared<IspDemosaicNoscale>());
    mgr->insert("Fcs", std::make_shared<IspFcs>());
    mgr->insert("Ee", std::make_shared<IspEe>());
    mgr->insert("Ldci", std::make_shared<IspLdci>());
    mgr->insert("Display", std::make_shared<IspDisplay>());
    mgr->insert("FcsDisplay", std::make_shared<IspFcsDisplay>());
    mgr->insert("EeLdci", std::make_shared<IspEeLdci>());
    mgr->insert("Vignetting", std::make_shared<IspVignetting>());
    mgr->insert("AutoContrast", std::make_shared<IspAutoContrast>());
    mgr->insert("Colorspace", std::make_shared<IspColorspace>());
    mgr->insert("WaveletDenoise", std::make_shared<IspWaveletDenoise>());
    mgr->insert("Bilateral", std::make_shared<IspBilateral>());
    return true;
}();

} // namespace
