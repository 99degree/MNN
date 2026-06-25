// GeometryExtra.cpp — GeometryComputer for OpType_Extra (pass-through)
#include "GeometryComputer.hpp"
using namespace MNN;

class GeometryExtra : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& cmd) const override {
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryExtra);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Extra});
}
REGISTER_GEOMETRY(GeometryExtra, _create);
