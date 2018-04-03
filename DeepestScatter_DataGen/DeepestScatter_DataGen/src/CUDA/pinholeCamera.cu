#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"

using namespace optix;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtBuffer<float4, 2>   resultBuffer;

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );

rtDeclareVariable(rtObject, objectRoot, , );
rtDeclareVariable(float3, errorColor, , );
rtDeclareVariable(float3, missColor, , );

rtDeclareVariable(unsigned int, radianceRayType, , );
rtDeclareVariable(float, sceneEPS, , );

RT_PROGRAM void pinholeCamera()
{
    size_t2 screen = resultBuffer.size();

    float2 d = make_float2(launchID) / make_float2(screen) * 2.f - 1.f;

    float3 origin = eye;
    float3 direction = normalize(d.x*U + d.y * V + W);

    optix::Ray ray(origin, direction, radianceRayType, sceneEPS);

    PerRayData_radiance prd;
    prd.importance = 1.0f;
    prd.depth = 0;

    rtTrace(objectRoot, ray, prd);
    resultBuffer[launchID] = make_float4(lerp(prd.result, missColor, prd.importance), 1);
}

rtDeclareVariable(PerRayData_radiance, resultRadiance, rtPayload, );

RT_PROGRAM void exception()
{
    resultBuffer[launchID] = make_float4(errorColor, 1);
}

RT_PROGRAM void miss()
{
    resultRadiance.result = missColor;
    resultRadiance.importance = 0;
}