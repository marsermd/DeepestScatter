#pragma once

#include <optixu/optixpp_namespace.h>

#include "SceneItem.h"

namespace DeepestScatter
{
    class Resources;

    class CloudPTRenderer: public SceneItem
    {
    public:
        CloudPTRenderer(optix::Context context, std::shared_ptr<Resources>& resources) :
            context(context),
            resources(resources) {}

        virtual ~CloudPTRenderer() override = default;

        void Init() override;
        void Update() override {} // Does nothing.

    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;

        optix::Material         cloudMaterial;

        optix::Geometry         geometry;
        optix::GeometryInstance geometryInstance;
        optix::GeometryGroup    geometryGroup;
        optix::Material         material;
    };
}