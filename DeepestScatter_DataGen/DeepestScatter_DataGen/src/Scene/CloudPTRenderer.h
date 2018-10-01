#pragma once

#include <optixu/optixpp_namespace.h>

#include "SceneItem.h"

namespace DeepestScatter
{
    class Resources;

    class CloudPTRenderer: public SceneItem
    {
    public:
        enum class RenderMode;

        CloudPTRenderer(RenderMode renderMode, optix::Context context, std::shared_ptr<Resources> resources):
            renderMode(renderMode),
            context(context),
            resources(resources) {}

        virtual ~CloudPTRenderer() override = default;

        void Init() override;
        void Reset() override {}
        void Update() override {}

        enum class RenderMode
        {
            Full,
            SunMultipleScatter
        };

    private:
        RenderMode renderMode;
        optix::Context context;
        std::shared_ptr<Resources> resources;

        optix::Material         cloudMaterial;

        optix::Geometry         geometry;
        optix::GeometryInstance geometryInstance;
        optix::GeometryGroup    geometryGroup;
        optix::Material         material;

        std::string getRenderProgramName();
    };
}