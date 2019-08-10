#pragma once

#include "ARenderer.h"

#pragma warning(push, 0)
#include <torch/script.h>
#pragma warning(pop)

namespace DeepestScatter
{
    class Resources;

    class DisneyRenderer : public ARenderer
    {
    public:
        DisneyRenderer(
            std::shared_ptr<optix::Context> context,
            std::shared_ptr<Resources> resources):
            context(*context.get()),
            resources(std::move(resources))
        {
        }

        virtual ~DisneyRenderer() = default;

        optix::Program getCamera() override;
        void init() override;
        void render(optix::Buffer frameResultBuffer) override;

        inline static const std::string NAME = "NN";
    private:
        void setupVariables(optix::Program& program);

        void renderRect(optix::uint2 start, optix::Buffer frameResultBuffer);

        std::shared_ptr<torch::jit::script::Module> module;

        optix::Context context;
        std::shared_ptr<Resources> resources;

        optix::Program clearRect;
        optix::Program blit;
        optix::Program camera;

        std::vector<torch::jit::IValue> networkInputs;

        optix::Buffer  networkInputBuffer;
        optix::Buffer  directRadianceBuffer;
        optix::Buffer  predictedRadianceBuffer;
    };
}
