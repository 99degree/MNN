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

// ── isp.ai::Aaf → isp.aaf Extra op ──
class IspAaf : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "aaf");
        return newVar->expr().first;
    }
};
// ── isp.ai::Ae → isp.ae Extra op ──
class IspAe : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "ae");
        return newVar->expr().first;
    }
};
// ── isp.ai::AfFocus → isp.affocus Extra op ──
class IspAfFocus : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "affocus");
        return newVar->expr().first;
    }
};
// ── isp.ai::AlgoGamma → isp.algogamma Extra op ──
class IspAlgoGamma : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "algogamma");
        return newVar->expr().first;
    }
};
// ── isp.ai::Awb → isp.awb Extra op ──
class IspAwb : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "awb");
        return newVar->expr().first;
    }
};
// ── isp.ai::BayerWb → isp.bayerwb Extra op ──
class IspBayerWb : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "bayerwb");
        return newVar->expr().first;
    }
};
// ── isp.ai::Blc → isp.blc Extra op ──
class IspBlc : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "blc");
        return newVar->expr().first;
    }
};
// ── isp.ai::Bnf → isp.bnf Extra op ──
class IspBnf : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "bnf");
        return newVar->expr().first;
    }
};
// ── isp.ai::CalibStats → isp.calibstats Extra op ──
class IspCalibStats : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "calibstats");
        return newVar->expr().first;
    }
};
// ── isp.ai::Ccm → isp.ccm Extra op ──
class IspCcm : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "ccm");
        return newVar->expr().first;
    }
};
// ── isp.ai::Cct → isp.cct Extra op ──
class IspCct : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "cct");
        return newVar->expr().first;
    }
};
// ── isp.ai::Ceh → isp.ceh Extra op ──
class IspCeh : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "ceh");
        return newVar->expr().first;
    }
};
// ── isp.ai::Cfa → isp.cfa Extra op ──
class IspCfa : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "cfa");
        return newVar->expr().first;
    }
};
// ── isp.ai::ChromaticAberration → isp.chromaticaberration Extra op ──
class IspChromaticAberration : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "chromaticaberration");
        return newVar->expr().first;
    }
};
// ── isp.ai::Cnf → isp.cnf Extra op ──
class IspCnf : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "cnf");
        return newVar->expr().first;
    }
};
// ── isp.ai::Crop → isp.crop Extra op ──
class IspCrop : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "crop");
        return newVar->expr().first;
    }
};
// ── isp.ai::CropBlc → isp.cropblc Extra op ──
class IspCropBlc : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "cropblc");
        return newVar->expr().first;
    }
};
// ── isp.ai::DebayerG2 → isp.debayerg2 Extra op ──
class IspDebayerG2 : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "debayerg2");
        return newVar->expr().first;
    }
};
// ── isp.ai::DemosaicEdge → isp.demosaicedge Extra op ──
class IspDemosaicEdge : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "demosaicedge");
        return newVar->expr().first;
    }
};
// ── isp.ai::DemosaicG2Ccm → isp.demosaicg2ccm Extra op ──
class IspDemosaicG2Ccm : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "demosaicg2ccm");
        return newVar->expr().first;
    }
};
// ── isp.ai::DisplayRgb → isp.displayrgb Extra op ──
class IspDisplayRgb : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "displayrgb");
        return newVar->expr().first;
    }
};
// ── isp.ai::DisplayUint8Argb → isp.displayuint8argb Extra op ──
class IspDisplayUint8Argb : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "displayuint8argb");
        return newVar->expr().first;
    }
};
// ── isp.ai::DisplayUint8Rgba → isp.displayuint8rgba Extra op ──
class IspDisplayUint8Rgba : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "displayuint8rgba");
        return newVar->expr().first;
    }
};
// ── isp.ai::DisplayUint8Bgra → isp.displayuint8bgra Extra op ──
class IspDisplayUint8Bgra : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "displayuint8bgra");
        return newVar->expr().first;
    }
};
// ── isp.ai::DisplayUint32Argb → isp.displayuint32argb Extra op ──
class IspDisplayUint32Argb : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "displayuint32argb");
        return newVar->expr().first;
    }
};
// ── isp.ai::DisplayUint32Rgba → isp.displayuint32rgba Extra op ──
class IspDisplayUint32Rgba : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "displayuint32rgba");
        return newVar->expr().first;
    }
};
// ── isp.ai::DisplayUint32Bgra → isp.displayuint32bgra Extra op ──
class IspDisplayUint32Bgra : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "displayuint32bgra");
        return newVar->expr().first;
    }
};
// ── isp.ai::Downscale → isp.downscale Extra op ──
class IspDownscale : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "downscale");
        return newVar->expr().first;
    }
};
// ── isp.ai::Dpc → isp.dpc Extra op ──
class IspDpc : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "dpc");
        return newVar->expr().first;
    }
};
// ── isp.ai::Gamma → isp.gamma Extra op ──
class IspGamma : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "gamma");
        return newVar->expr().first;
    }
};
// ── isp.ai::GpuWarp → isp.gpuwarp Extra op ──
class IspGpuWarp : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "gpuwarp");
        return newVar->expr().first;
    }
};
// ── isp.ai::HdrMerge → isp.hdrmerge Extra op ──
class IspHdrMerge : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "hdrmerge");
        return newVar->expr().first;
    }
};
// ── isp.ai::Histogram → isp.histogram Extra op ──
class IspHistogram : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "histogram");
        return newVar->expr().first;
    }
};
// ── isp.ai::IspcStats → isp.ispcstats Extra op ──
class IspIspcStats : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "ispcstats");
        return newVar->expr().first;
    }
};
// ── isp.ai::LocalContrast → isp.localcontrast Extra op ──
class IspLocalContrast : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "localcontrast");
        return newVar->expr().first;
    }
};
// ── isp.ai::Lsc → isp.lsc Extra op ──
class IspLsc : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "lsc");
        return newVar->expr().first;
    }
};
// ── isp.ai::Nlm → isp.nlm Extra op ──
class IspNlm : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "nlm");
        return newVar->expr().first;
    }
};
// ── isp.ai::Normalize → isp.normalize Extra op ──
class IspNormalize : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "normalize");
        return newVar->expr().first;
    }
};
// ── isp.ai::Pyramid → isp.pyramid Extra op ──
class IspPyramid : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "pyramid");
        return newVar->expr().first;
    }
};
// ── isp.ai::RawBlc → isp.rawblc Extra op ──
class IspRawBlc : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "rawblc");
        return newVar->expr().first;
    }
};
// ── isp.ai::Saturation → isp.saturation Extra op ──
class IspSaturation : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "saturation");
        return newVar->expr().first;
    }
};
// ── isp.ai::Stats → isp.stats Extra op ──
class IspStats : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "stats");
        return newVar->expr().first;
    }
};
// ── isp.ai::TemporalDenoise → isp.temporaldenoise Extra op ──
class IspTemporalDenoise : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "temporaldenoise");
        return newVar->expr().first;
    }
};
// ── isp.ai::Tone → isp.tone Extra op ──
class IspTone : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "tone");
        return newVar->expr().first;
    }
};
// ── isp.ai::ToneStats → isp.tonestats Extra op ──
class IspToneStats : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "tonestats");
        return newVar->expr().first;
    }
};
// ── isp.ai::Unsharp → isp.unsharp Extra op ──
class IspUnsharp : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "unsharp");
        return newVar->expr().first;
    }
};
// ── isp.ai::Warp → isp.warp Extra op ──
class IspWarp : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "warp");
        return newVar->expr().first;
    }
};
// ── isp.ai::YuvBcc → isp.yuvbcc Extra op ──
class IspYuvBcc : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "yuvbcc");
        return newVar->expr().first;
    }
};
// ── isp.ai::YuvSat → isp.yuvsat Extra op ──
class IspYuvSat : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "yuvsat");
        return newVar->expr().first;
    }
};
// ── isp.ai::ZoneStats → isp.zonestats Extra op ──
class IspZoneStats : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        auto newVar = createVulkanFuseOp(expr, op, "zonestats");
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
    mgr->insert("Aaf", std::make_shared<IspAaf>());
    mgr->insert("Ae", std::make_shared<IspAe>());
    mgr->insert("AfFocus", std::make_shared<IspAfFocus>());
    mgr->insert("AlgoGamma", std::make_shared<IspAlgoGamma>());
    mgr->insert("Awb", std::make_shared<IspAwb>());
    mgr->insert("BayerWb", std::make_shared<IspBayerWb>());
    mgr->insert("Blc", std::make_shared<IspBlc>());
    mgr->insert("Bnf", std::make_shared<IspBnf>());
    mgr->insert("CalibStats", std::make_shared<IspCalibStats>());
    mgr->insert("Ccm", std::make_shared<IspCcm>());
    mgr->insert("Cct", std::make_shared<IspCct>());
    mgr->insert("Ceh", std::make_shared<IspCeh>());
    mgr->insert("Cfa", std::make_shared<IspCfa>());
    mgr->insert("ChromaticAberration", std::make_shared<IspChromaticAberration>());
    mgr->insert("Cnf", std::make_shared<IspCnf>());
    mgr->insert("Crop", std::make_shared<IspCrop>());
    mgr->insert("CropBlc", std::make_shared<IspCropBlc>());
    mgr->insert("DebayerG2", std::make_shared<IspDebayerG2>());
    mgr->insert("DemosaicEdge", std::make_shared<IspDemosaicEdge>());
    mgr->insert("DemosaicG2Ccm", std::make_shared<IspDemosaicG2Ccm>());
    mgr->insert("DisplayRgb", std::make_shared<IspDisplayRgb>());
    mgr->insert("DisplayUint8Argb", std::make_shared<IspDisplayUint8Argb>());
    mgr->insert("DisplayUint8Rgba", std::make_shared<IspDisplayUint8Rgba>());
    mgr->insert("DisplayUint8Bgra", std::make_shared<IspDisplayUint8Bgra>());
    mgr->insert("DisplayUint32Argb", std::make_shared<IspDisplayUint32Argb>());
    mgr->insert("DisplayUint32Rgba", std::make_shared<IspDisplayUint32Rgba>());
    mgr->insert("DisplayUint32Bgra", std::make_shared<IspDisplayUint32Bgra>());
    mgr->insert("Downscale", std::make_shared<IspDownscale>());
    mgr->insert("Dpc", std::make_shared<IspDpc>());
    mgr->insert("Gamma", std::make_shared<IspGamma>());
    mgr->insert("GpuWarp", std::make_shared<IspGpuWarp>());
    mgr->insert("HdrMerge", std::make_shared<IspHdrMerge>());
    mgr->insert("Histogram", std::make_shared<IspHistogram>());
    mgr->insert("IspcStats", std::make_shared<IspIspcStats>());
    mgr->insert("LocalContrast", std::make_shared<IspLocalContrast>());
    mgr->insert("Lsc", std::make_shared<IspLsc>());
    mgr->insert("Nlm", std::make_shared<IspNlm>());
    mgr->insert("Normalize", std::make_shared<IspNormalize>());
    mgr->insert("Pyramid", std::make_shared<IspPyramid>());
    mgr->insert("RawBlc", std::make_shared<IspRawBlc>());
    mgr->insert("Saturation", std::make_shared<IspSaturation>());
    mgr->insert("Stats", std::make_shared<IspStats>());
    mgr->insert("TemporalDenoise", std::make_shared<IspTemporalDenoise>());
    mgr->insert("Tone", std::make_shared<IspTone>());
    mgr->insert("ToneStats", std::make_shared<IspToneStats>());
    mgr->insert("Unsharp", std::make_shared<IspUnsharp>());
    mgr->insert("Warp", std::make_shared<IspWarp>());
    mgr->insert("YuvBcc", std::make_shared<IspYuvBcc>());
    mgr->insert("YuvSat", std::make_shared<IspYuvSat>());
    mgr->insert("ZoneStats", std::make_shared<IspZoneStats>());
    return true;
}();

} // namespace
