#pragma once

#include <optixu/optixpp_namespace.h>
#include <utility>

#include "Scene/SceneItem.h"

namespace DeepestScatter
{
    class Resources;

    class RadianceCollector : public SceneItem
    {
    public:
        RadianceCollector(optix::Context context, std::shared_ptr<Resources> resources):
            context(context),
            resources(std::move(resources)) {}

        virtual ~RadianceCollector() override = default;

        void init() override;
        void reset() override {}
        void update() override {} // Does nothing.

    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;
    };
}