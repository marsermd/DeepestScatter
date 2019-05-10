#pragma once

#include "ARenderer.h"

#pragma warning(push, 0)
#include <torch/script.h>
#pragma warning(pop)

namespace DeepestScatter
{
    class Resources;

    class BakedRenderer : public ARenderer
    {
    public:
        BakedRenderer(
            std::shared_ptr<optix::Context> context,
            std::shared_ptr<Resources> resources) :
            context(*context.get()),
            resources(std::move(resources))
        {
        }

        virtual ~BakedRenderer() = default;

        optix::Program getCamera() override;
        void init() override;
        void render(optix::Buffer frameResultBuffer) override;

    private:

        class Baker
        {
        public:
            Baker(optix::Context context, std::shared_ptr<Resources> resources);

            optix::Buffer bake();
        
        private:
            void bakeAtZ(uint32_t posZ);

            optix::Context context;
            std::shared_ptr<Resources> resources;

            std::shared_ptr<torch::jit::script::Module> lightProbeModel;
            std::vector<torch::jit::IValue> lightProbeInputs;

            optix::Program collect;
            optix::Buffer descriptors;
            optix::Buffer result;
        };

        void setupVariables(optix::Program& program);

        void renderRect(optix::uint2 start, optix::Buffer frameResultBuffer);

        std::shared_ptr<torch::jit::script::Module> renderModel;

        optix::Context context;
        std::shared_ptr<Resources> resources;

        optix::Program camera;

        std::vector<torch::jit::IValue> rendererInputs;

        optix::Buffer bakedLightProbes;

        optix::Buffer rendererInputBuffer;
        optix::Buffer directRadianceBuffer;
    };
}
