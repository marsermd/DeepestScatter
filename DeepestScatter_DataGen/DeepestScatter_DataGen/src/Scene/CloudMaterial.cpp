#include "CloudMaterial.h"
#include "Util/Resources.h"

#include "CUDA/rayData.cuh"
#include <iostream>

namespace DeepestScatter
{
    CloudMaterial::CloudMaterial(Settings settings, std::shared_ptr<optix::Context> context, std::shared_ptr<Resources> resources)
        : context(*context.get()),
          resources(std::move(resources)),
          renderSettings(settings)
    {
        this->context["sampleStep"]->setFloat(settings->sampleStep);
    }

    void CloudMaterial::init()
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
        material->setClosestHitProgram(
            DisneyDescriptorRayData::rayId,
            resources->loadProgram("disneyDescriptorMaterial.cu", "sampleDisneyDescriptor"));

        geometryInstance = context->createGeometryInstance(geometry, &material, &material + 1);
        geometryGroup = context->createGeometryGroup();
        geometryGroup->addChild(geometryInstance);
        geometryGroup->setAcceleration(context->createAcceleration("MedianBvh", "Bvh"));

        context["objectRoot"]->set(geometryGroup);
    }

    std::string CloudMaterial::getRenderProgramName() const
    {
        switch (renderSettings->mode)
        {
        case Cloud::Rendering::Mode::SunAndSkyAllScatter:
            return "totalRadiance";
        case Cloud::Rendering::Mode::SunMultipleScatter:
            return "multipleScatterSunRadiance";
        case Cloud::Rendering::Mode::SunSingleScatter:
            return "singleScatterSunRadiance";
        default:
            throw std::invalid_argument("Invalid Render Mode");
        }
    }
}
