#include "CloudPTRenderer.h"
#include "Util/Resources.h"

#include "CUDA/rayData.cuh"

namespace DeepestScatter
{
    void CloudPTRenderer::Init()
    {
        geometry = context->createGeometry();
        geometry->setBoundingBoxProgram(resources->loadProgram("cloudBBox.cu", "bounds"));
        geometry->setIntersectionProgram(resources->loadProgram("cloudBBox.cu", "intersect"));
        geometry->setPrimitiveCount(1u);
        geometry["minimalRayDistance"]->setFloat(0.001f);

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

    std::string CloudPTRenderer::getRenderProgramName()
    {
        switch (renderMode)
        {
        case RenderMode::Full:
            return "totalRadiance";
        case RenderMode::SunMultipleScatter:
            return "multipleScatterSunRadiance";
        default:
            throw std::invalid_argument("Invalid Render Mode");
        }
    }
}