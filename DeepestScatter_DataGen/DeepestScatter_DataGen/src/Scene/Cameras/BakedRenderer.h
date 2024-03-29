#pragma once

#include "ARenderer.h"

#pragma warning(push, 0)
#include <torch/script.h>
#include <optix_sizet.h>
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

        inline static const std::string NAME = "BNN";
    private:

        class Baker
        {
        public:
            Baker(optix::Context context, std::shared_ptr<Resources> resources, optix::size_t3 probeCount);

            optix::Buffer bake();
        
        private:
            void bakeAtZ(uint32_t posZ);

            optix::Context context;
            std::shared_ptr<Resources> resources;
            optix::size_t3 probeCount;

            std::shared_ptr<torch::jit::script::Module> lightProbeModel;
            std::vector<torch::jit::IValue> lightProbeInputs;

            optix::Program collect;
            optix::Buffer descriptors;
            optix::Buffer result;
        };

        void setupVariables(optix::Program& program);

        void renderRect(optix::uint2 start);

        std::shared_ptr<torch::jit::script::Module> renderModel;

        optix::Context context;
        std::shared_ptr<Resources> resources;

        optix::Program camera;
        optix::Program blit;

        std::vector<torch::jit::IValue> rendererInputs;

        optix::size_t3 lightProbeCount;
        optix::Buffer bakedLightProbes;

        optix::Buffer lightProbeInputBuffer;
        optix::Buffer descriptorInputBuffer;
        optix::Buffer directRadianceBuffer;
        optix::Buffer predictedRadianceBuffer;
    };
}
