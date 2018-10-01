#pragma once

#include <optixu/optixpp_namespace.h>

#include "SceneItem.h"

namespace DeepestScatter
{
    class Resources;

    class RadianceCollector : public SceneItem
    {
    public:
        RadianceCollector(optix::Context context, std::shared_ptr<Resources> resources):
            context(context),
            resources(resources) {}

        virtual ~RadianceCollector() override = default;

        void Init() override;
        void Reset() override {}
        void Update() override {} // Does nothing.

    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;
    };
}