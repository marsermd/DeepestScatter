#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"
#include "random.cuh"

using namespace optix;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtBuffer<DeepestScatter::Gpu::DisneyNetworkInput, 2> networkInputBuffer;
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

    DisneyDescriptorRayData prd;
    prd.descriptor = DeepestScatter::Gpu::DisneyDescriptor();
    prd.intersectionInfo.hasScattered = false;
    prd.intersectionInfo.radiance = make_float3(0);

    optix::Ray ray(origin, direction, prd.rayId, sceneEPS);
    rtTrace(objectRoot, ray, prd);

    float angle = acos(dot(lightDirection, direction));

    networkInputBuffer[launchID].fill(prd.descriptor, angle);
    directRadianceBuffer[launchID] = prd.intersectionInfo;
}

RT_PROGRAM void clearRect()
{
    //todo: proably uncomment these. Right now it jut makes performance worse.
    //networkInputBuffer[launchID].clear();
    //directRadianceBuffer[launchID].radiance = make_float3(0);
    //directRadianceBuffer[launchID].hasScattered = false;
}