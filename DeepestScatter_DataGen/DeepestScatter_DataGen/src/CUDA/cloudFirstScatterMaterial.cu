#include "cloud.cuh"

#include "rayData.cuh"

rtDeclareVariable(ScatteringRayData, firstScatter, rtPayload, );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

RT_PROGRAM void firstScatterPosition()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y, subframeId);

    ScatteringEvent scatter = getNextScatteringEvent(seed, pos, direction);
    if (isInBox(scatter.scatterPos))
    {
        firstScatter.position = scatter.scatterPos - 0.5f * bboxSize;
        firstScatter.hasScattered = true;
    }
    else
    {
        firstScatter.hasScattered = false;
    }
}