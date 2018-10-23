#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"
#include "random.cuh"
#include <cassert>

using namespace optix;

rtDeclareVariable(uint, launchID, rtLaunchIndex, );

rtBuffer<float3, 1> directionBuffer;
rtBuffer<float3, 1> positionBuffer;

rtBuffer<float, 1> resultBuffer;
rtBuffer<float, 1> varianceBuffer;

rtDeclareVariable(rtObject, objectRoot, , );
rtDeclareVariable(float3, errorColor, , );

rtDeclareVariable(float, sceneEPS, , );
rtDeclareVariable(unsigned int, subframeId, , );


RT_PROGRAM void estimateEmission()
{
    float N = subframeId;
    if (subframeId > 100 && 1.69f * sqrt(varianceBuffer[launchID] / N) / resultBuffer[launchID] / sqrtf(N) < 0.02f)
    {
        return;
    }

    float3 origin = positionBuffer[launchID];
    float3 direction = normalize(directionBuffer[launchID]);

    RadianceRayData prd;
    prd.result = make_float3(0);
    prd.importance = 1;
    optix::Ray ray(origin, direction, prd.rayId, sceneEPS);

    rtTrace(objectRoot, ray, prd);

    float newResult = prd.result.x;
    float newWeight = 1.0f / (float)subframeId;

    float previousMean = resultBuffer[launchID];
    float newMean = resultBuffer[launchID] + (newResult - previousMean) * newWeight;
    resultBuffer[launchID] = newMean;

    varianceBuffer[launchID] = varianceBuffer[launchID] + (newResult - previousMean) * (newResult - newMean);
}

RT_PROGRAM void clear()
{
    resultBuffer[launchID] = 0;
    varianceBuffer[launchID] = 0;
}