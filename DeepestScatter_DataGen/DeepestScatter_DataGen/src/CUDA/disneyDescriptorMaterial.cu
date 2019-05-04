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

    float3 direction = normalize(ray.direction);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    ScatteringEvent scatter = getNextScatteringEvent(seed, pos, direction);

    if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
    {
        resultDescriptor.intersectionInfo.hasScattered = false;
    }
    else
    {
        resultDescriptor.intersectionInfo.hasScattered = true;
        //todo:resultDescriptor.intersectionInfo.radiance = getInScattering(scatter, direction, false);
        setupDisneyDescriptor(resultDescriptor.descriptor, scatter.scatterPos, direction);
    }

    //todo:
    int depth = 0;
    while (isInBox(pos))
    {
        depth++;
        if (depth == 1000)
        {
            break;
        }

        ScatteringEvent scatter = getNextScatteringEvent(seed, pos, direction);

        if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
        {
            break;
        }
        else
        {
            if (depth > 1)
            {
                // next event estimation
                resultDescriptor.intersectionInfo.radiance += getInScattering(scatter, direction, true);
            }

            pos = scatter.scatterPos;

            direction = getNewDirection(seed, direction);
        }
    }
}