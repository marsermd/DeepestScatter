#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"
#include "PointRadianceTask.h"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint, launchID, rtLaunchIndex, );

rtBuffer<PointRadianceTask, 1> tasks;

rtDeclareVariable(rtObject, objectRoot, , );

rtDeclareVariable(float, sceneEPS, , );
rtDeclareVariable(unsigned int, subframeId, , );


RT_PROGRAM void estimateEmission()
{
    const float3 origin = tasks[launchID].position;
    const float3 direction = tasks[launchID].direction;

    RadianceRayData prd{};
    prd.result = make_float3(0);
    prd.importance = 1;
    optix::Ray ray(origin, direction, prd.rayId, sceneEPS);

    rtTrace(objectRoot, ray, prd);

    tasks[launchID].addExperimentResult(prd.result.x);
}

RT_PROGRAM void clear()
{
    tasks[launchID].radiance = 0;
    tasks[launchID].runningVariance = 0;
    tasks[launchID].experimentCount = 0;
}