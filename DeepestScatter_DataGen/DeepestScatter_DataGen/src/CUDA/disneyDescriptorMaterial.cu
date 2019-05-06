/*
* Only the direct light of the sun.
*/
#include "cloud.cuh"
#include "rayData.cuh"
#include <optix_device.h>
#include "DisneyDescriptor.cuh"

rtDeclareVariable(DisneyDescriptorRayData, resultDescriptor, rtPayload, );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

RT_PROGRAM void sampleDisneyDescriptor()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 pos = hitPoint;

    const float3 direction = normalize(ray.direction);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    float jitter = rnd(seed) / 100;
    float opticalDistance = (subframeId % 100) / 100.0f + jitter;

    const ScatteringEvent scatter = getNextScatteringEvent(opticalDistance, pos, direction);

    if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
    {
        resultDescriptor.intersectionInfo.hasScattered = false;
    }
    else
    {
        resultDescriptor.intersectionInfo.hasScattered = true;
        resultDescriptor.intersectionInfo.radiance = getInScattering(scatter, direction, false);
        setupDisneyDescriptor(resultDescriptor.descriptor, scatter.scatterPos - 0.5f * bboxSize, direction);
    }

}