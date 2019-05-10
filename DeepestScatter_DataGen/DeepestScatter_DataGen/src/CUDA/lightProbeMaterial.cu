/*
* Only the direct light of the sun.
*/
#include "cloud.cuh"
#include "rayData.cuh"
#include <optix_device.h>
#include "LightProbe.h"

using namespace DeepestScatter::Gpu;

rtDeclareVariable(LightProbeRayData, rayData, rtPayload, );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtBuffer<LightProbe, 3> bakedLightProbes;
rtDeclareVariable(float, cloudSizeInMeters, , );

static __host__ __device__ __inline__ uint getId(float x, float step)
{
    return static_cast<uint>(round(x * step));
}

static __host__ __device__ __inline__ optix::uint3 getId(optix::float3 v, float step)
{
    return optix::make_uint3(
        getId(v.x, step),
        getId(v.y, step),
        getId(v.z, step)
    );
}

RT_PROGRAM void sampleLightProbe()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 pos = hitPoint;

    const float3 direction = normalize(ray.direction);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    //const uint step = 50;
    //const float jitter = rnd(seed) / step;
    //float opticalDistance = (subframeId % step) * 1.0f / step + jitter;

    const ScatteringEvent scatter = getNextScatteringEvent(seed, pos, direction);

    if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
    {
        rayData.intersectionInfo.hasScattered = false;
    }
    else
    {
        rayData.intersectionInfo.hasScattered = true;
        rayData.intersectionInfo.radiance = getInScattering(scatter, direction, false);

        const float3 eZ1 = normalize(lightDirection);
        const float3 eX1 = normalize(cross(lightDirection, direction));
        const float3 eY1 = cross(eX1, eZ1);

        const float3 eZ2 = normalize(lightDirection);
        const float3 eX2 = normalize(cross(lightDirection, make_float3(0, 0, 1)));
        const float3 eY2 = cross(eX2, eZ2);
        rayData.lightProbe.omega = acos(dot(lightDirection, direction));
        rayData.lightProbe.alpha = acos(dot(eY1, eY2));

        uint3 lightProbeId = getId(scatter.scatterPos, 75);
        rayData.lightProbe.lightProbe = bakedLightProbes[lightProbeId];

        float3 lightProbePos = make_float3(
            lightProbeId.x / 75.f,
            lightProbeId.y / 75.f,
            lightProbeId.z / 75.f
        );

        float3 offset = (scatter.scatterPos - lightProbePos) * cloudSizeInMeters;

        rayData.lightProbe.offset = make_float3(
            dot(offset, eX2),
            dot(offset, eY2),
            dot(offset, eZ2)
        );
    }

}