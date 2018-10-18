#pragma once

#include <optixu/optixpp_namespace.h>
#include <memory>

#include "Scene/SceneItem.h"
#include "Scene/SceneDescription.h"

namespace DeepestScatter
{
    class Resources;

    class CloudPTRenderer: public SceneItem
    {
    public:
        typedef Cloud::Rendering Settings;

        CloudPTRenderer(Settings settings, optix::Context context, std::shared_ptr<Resources> resources);

        void init() override;
        void reset() override {}
        void update() override {}

    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;
        Settings renderSettings;

        optix::Material         cloudMaterial;

        optix::Geometry         geometry;
        optix::GeometryInstance geometryInstance;
        optix::GeometryGroup    geometryGroup;
        optix::Material         material;

        std::string getRenderProgramName() const;
    };
}