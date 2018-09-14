#include "CloudPTRenderer.h"
#include "Util/Resources.h"

void DeepestScatter::CloudPTRenderer::Init()
{
    geometry = context->createGeometry();
    geometry->setBoundingBoxProgram(resources->loadProgram("cloud.cu", "bounds"));
    geometry->setIntersectionProgram(resources->loadProgram("cloud.cu", "intersect"));
    geometry->setPrimitiveCount(1u);
    geometry["minimalRayDistance"]->setFloat(0.001f);

    material = context->createMaterial();
    material->setClosestHitProgram(0, resources->loadProgram("cloud.cu", "closestHitRadiance"));

    geometryInstance = context->createGeometryInstance(geometry, &material, &material + 1);
    geometryGroup = context->createGeometryGroup();
    geometryGroup->addChild(geometryInstance);
    geometryGroup->setAcceleration(context->createAcceleration("MedianBvh", "Bvh"));

    context["objectRoot"]->set(geometryGroup);
}
