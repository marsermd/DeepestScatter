#pragma once

#include <optixu/optixpp_namespace.h>
#include <utility>
#include <memory>

#include "Scene/SceneItem.h"

namespace DeepestScatter
{
    class Resources;

    class RadianceCollector : public SceneItem
    {
    public:
        RadianceCollector(std::shared_ptr<optix::Context> context, std::shared_ptr<Resources> resources):
            context(*context.get()),
            resources(std::move(resources)) {}

        void init() override;
        void reset() override {}
        void update() override {} // Does nothing.

    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;
    };
}