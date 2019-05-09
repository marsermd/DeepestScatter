#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtDeclareVariable(unsigned int, subframeId, , );
rtDeclareVariable(float3, errorColor, , );

rtBuffer<float4, 2> frameResultBuffer;

rtBuffer<float4, 2> progressiveBuffer;
rtBuffer<float4, 2> varianceBuffer;

RT_PROGRAM void updateFrameResult()
{
    float newWeight = 1.0f / (float)subframeId;

    float4 newResult = frameResultBuffer[launchID];
    float4 previousMean = progressiveBuffer[launchID];
    float4 newMean = progressiveBuffer[launchID] + (newResult - previousMean) * newWeight;
    progressiveBuffer[launchID] = newMean;
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