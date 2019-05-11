#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"
#include "random.cuh"
#include "LightProbe.h"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtBuffer<LightProbeRendererInput, 2> lightProbeInputBuffer;
rtBuffer<BakedRendererDescriptor, 2> descriptorInputBuffer;
rtBuffer<IntersectionInfo, 2> directRadianceBuffer;
rtBuffer<float4, 2> frameResultBuffer;

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );

rtDeclareVariable(float3, lightDirection, , );

rtDeclareVariable(uint2, rectOrigin, , );

rtDeclareVariable(rtObject, objectRoot, , );

rtDeclareVariable(float, sceneEPS, , );
rtDeclareVariable(unsigned int, subframeId, , );

RT_PROGRAM void pinholeCamera()
{
    size_t2 screen = frameResultBuffer.size();

    float2 d = make_float2(launchID + rectOrigin) / make_float2(screen) * 2.f - 1.f;

    uint32_t seed = tea<3>(subframeId);
    float2 jitter = (make_float2(rnd(seed), rnd(seed)) * 2 - 1) / make_float2(screen); // todo;
    d += jitter;

    float3 origin = eye;
    float3 direction = normalize(d.x*U + d.y * V + W);

    LightProbeRayData prd;
    prd.intersectionInfo = &directRadianceBuffer[launchID];
    prd.lightProbe = &lightProbeInputBuffer[launchID];
    prd.descriptor = &descriptorInputBuffer[launchID];
    prd.intersectionInfo->hasScattered = false;
    prd.intersectionInfo->radiance = make_float3(0);

    optix::Ray ray(origin, direction, prd.rayId, sceneEPS);
    rtTrace(objectRoot, ray, prd);
}