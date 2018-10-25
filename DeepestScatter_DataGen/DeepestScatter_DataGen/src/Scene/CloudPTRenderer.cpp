#include "CloudPTRenderer.h"
#include "Util/Resources.h"

#include "CUDA/rayData.cuh"
#include <iostream>

namespace DeepestScatter
{
    CloudPTRenderer::CloudPTRenderer(Settings settings, std::shared_ptr<optix::Context> context, std::shared_ptr<Resources> resources)
        : context(*context.get()),
          resources(std::move(resources)),
          renderSettings(settings)
    {
        this->context["sampleStep"]->setFloat(settings->sampleStep);
    }

    void CloudPTRenderer::init()
    {
        geometry = context->createGeometry();
        geometry->setBoundingBoxProgram(resources->loadProgram("cloudBBox.cu", "bounds"));
        geometry->setIntersectionProgram(resources->loadProgram("cloudBBox.cu", "intersect"));
        geometry->setPrimitiveCount(1u);
        geometry["minimalRayDistance"]->setFloat(0.000001f);

        material = context->createMaterial();
        material->setClosestHitProgram(
            RadianceRayData::rayId,
            resources->loadProgram("cloudRadianceMaterials.cu", getRenderProgramName()));
        material->setClosestHitProgram(
            ScatteringRayData::rayId,
            resources->loadProgram("cloudFirstScatterMaterial.cu", "firstScatterPosition"));

        geometryInstance = context->createGeometryInstance(geometry, &material, &material + 1);
        geometryGroup = context->createGeometryGroup();
        geometryGroup->addChild(geometryInstance);
        geometryGroup->setAcceleration(context->createAcceleration("MedianBvh", "Bvh"));

        context["objectRoot"]->set(geometryGroup);
    }

    std::string CloudPTRenderer::getRenderProgramName() const
    {
        switch (renderSettings->mode)
        {
        case Cloud::Rendering::Mode::Full:
            return "totalRadiance";
        case Cloud::Rendering::Mode::SunMultipleScatter:
            return "multipleScatterSunRadiance";
        default:
            throw std::invalid_argument("Invalid Render Mode");
        }
    }
}
