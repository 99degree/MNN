// GeometryExtra.cpp — GeometryComputer for OpType_Extra (pass-through directly to backend)
#include "GeometryComputer.hpp"
#include "core/Command.hpp"
namespace MNN {

class GeometryExtra : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& cmd) const override {
        // Create a command that passes the original Extra op directly to the backend.
        // This allows VulkanFuse (or any backend's Extra handler) to execute the op.
        auto command = std::make_shared<Command>();
        command->op = op;
        command->inputs = inputs;
        command->outputs = outputs;
        cmd.command.emplace_back(std::move(command));
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryExtra);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Extra});
}
REGISTER_GEOMETRY(GeometryExtra, _create);

} // namespace MNN
