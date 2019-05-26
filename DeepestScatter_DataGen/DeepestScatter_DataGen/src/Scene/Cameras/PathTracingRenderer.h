#pragma once
#include "ARenderer.h"

namespace DeepestScatter
{
    class Resources;

    class PathTracingRenderer : public ARenderer
    {
    public:
        PathTracingRenderer(
            std::shared_ptr<optix::Context> context,
            std::shared_ptr<Resources> resources) :
            context(*context.get()),
            resources(std::move(resources))
        {
        }

        virtual ~PathTracingRenderer() = default;

        optix::Program getCamera() override;
        void init() override;
        void render(optix::Buffer frameResultBuffer) override;

    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;

        optix::Program camera;
    };
}
