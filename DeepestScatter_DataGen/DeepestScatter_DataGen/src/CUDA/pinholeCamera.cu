#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"

using namespace optix;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtBuffer<float4, 2>   progressiveBuffer;

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );

rtDeclareVariable(rtObject, objectRoot, , );
rtDeclareVariable(float3, errorColor, , );
rtDeclareVariable(float3, missColor, , );

rtDeclareVariable(float, sceneEPS, , );
rtDeclareVariable(unsigned int, radianceRayType, , );
rtDeclareVariable(unsigned int, subframeId, , );

RT_PROGRAM void pinholeCamera()
{
    size_t2 screen = progressiveBuffer.size();

    float2 d = make_float2(launchID) / make_float2(screen) * 2.f - 1.f;

    float3 origin = eye;
    float3 direction = normalize(d.x*U + d.y * V + W);

    optix::Ray ray(origin, direction, radianceRayType, sceneEPS);

    PerRayData_radiance prd;
    prd.importance = 1.0f;
    prd.depth = 0;

    rtTrace(objectRoot, ray, prd); 

    float newWeight = 1.0f / (float)subframeId;
    float oldWeight = 1.0f - newWeight;

    float4 newResult = make_float4(prd.result, 1);

    progressiveBuffer[launchID] = oldWeight * progressiveBuffer[launchID] + newWeight * newResult;
}

RT_PROGRAM void clearScreen()
{
    progressiveBuffer[launchID] = make_float4(0, 0, 0, 1);
}

RT_PROGRAM void exception()
{
    progressiveBuffer[launchID] = make_float4(errorColor, 1);
}

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, resultRadiance, rtPayload, );
rtDeclareVariable(float3, skyIntensity, , );
rtDeclareVariable(float3, groundIntensity, , );
rtDeclareVariable(float, lightIntensity, , );
rtDeclareVariable(float3, lightColor, , );
rtDeclareVariable(float3, lightDirection, , );

RT_PROGRAM void miss()
{
    float3 direction = normalize(ray.direction);
    float3 normalizedLightDirection = normalize(lightDirection);

    float cosLightAngle = dot(-normalizedLightDirection, direction);
    float3 currentLight = make_float3(0);

    if (cosLightAngle > 0.99998930414f) // cos(0.53 / 180 * pi / 2)
    {
        currentLight = lightColor * lightIntensity;
    }
    else
    {
        float t = clamp((direction.y + 0.5f) / 1.5f, 0.f, 1.f);
        currentLight = lerp(groundIntensity, skyIntensity, t);
    }

    resultRadiance.result = currentLight;
    resultRadiance.importance = 0;
}