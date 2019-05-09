#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"

using namespace optix;

rtDeclareVariable(unsigned int, subframeId, , );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtBuffer<float4, 2> frameResultBuffer;
rtBuffer<float4, 2> progressiveBuffer;
rtBuffer<float4, 2> varianceBuffer;

rtDeclareVariable(float3, errorColor, , );

RT_PROGRAM void updateFrameResult()
{
    const float newWeight = 1.0f / static_cast<float>(subframeId);

    const float4 newResult = frameResultBuffer[launchID];
    const float4 previousMean = progressiveBuffer[launchID];
    const float4 newMean = progressiveBuffer[launchID] + (newResult - previousMean) * newWeight;
    progressiveBuffer[launchID] = newMean;

    varianceBuffer[launchID] = varianceBuffer[launchID] + (newResult - previousMean) * (newResult - newMean);
}

RT_PROGRAM void clearScreen()
{
    frameResultBuffer[launchID] = make_float4(0);
    progressiveBuffer[launchID] = make_float4(0);
    varianceBuffer[launchID] = make_float4(0);
}

RT_PROGRAM void exception()
{
    progressiveBuffer[launchID] = make_float4(errorColor, 1);
}

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(DisneyDescriptorRayData, resultRadiance, rtPayload, );

RT_PROGRAM void miss()
{
}