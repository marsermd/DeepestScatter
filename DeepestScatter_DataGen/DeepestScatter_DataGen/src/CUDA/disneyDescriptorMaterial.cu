/*
* Only the direct light of the sun.
*/
#include "cloud.cuh"
#include "rayData.cuh"
#include <optix_device.h>
#include "DisneyDescriptor.cuh"

using namespace DeepestScatter::Gpu;

rtDeclareVariable(DisneyDescriptorRayData, rayData, rtPayload, );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

RT_PROGRAM void sampleDisneyDescriptor()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 pos = hitPoint;

    const float3 direction = normalize(ray.direction);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    //const uint step = 50;
    //const float jitter = rnd(seed) / step;
    //float opticalDistance = (subframeId % step) * 1.0f / step + jitter;

    const float transmittance = getNextScatteringEvent(seed, pos, direction, false).transmittance;
    const ScatteringEvent scatter = getNextScatteringEvent(1 - rnd(seed) * (1 - transmittance), pos, direction);

    rayData.intersectionInfo->transmittance = transmittance;

    if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
    {
        rayData.intersectionInfo->hasScattered = false;
        rayData.intersectionInfo->radiance = make_float3(0);
    }
    else
    {
        rayData.intersectionInfo->hasScattered = true;
        rayData.intersectionInfo->radiance = getInScattering(scatter, direction, false);
        setupHierarchicalDescriptor<DisneyNetworkInput, float>(
            *rayData.descriptor, scatter.scatterPos - 0.5f * bboxSize, direction);
    }

}