#include "cloud.cuh"

rtDeclareVariable(RadianceRayData, resultRadiance, rtPayload, );

RT_PROGRAM void totalRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 radiance = make_float3(0);
    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    resultRadiance.result = make_float3(0);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y, subframeId);

    float skySampleProbability = 0.1f;
    bool shouldSampleSky = subframeId % 10 == 0;

    int depth = 0;
    while (isInBox(pos))
    {
        depth++;
        if (depth == 1000)
        {
            break;
        }

        ScatteringEvent scatter = getNextScatteringEvent(seed, pos, direction, !shouldSampleSky);

        if (shouldSampleSky)
        {
            // We check that depth equals one because for all other depths, 
            // the light from the sun is already taken into account by next event estimation
            if (depth == 1)
            {
                radiance += sampleSun(direction) / skySampleProbability;
            }
            radiance += sampleSky(scatter, direction) / skySampleProbability;
        }

        if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
        {
            break;
        }
        else
        {
            // next event estimation
            radiance += getInScattering(scatter, direction, depth != 1);

            pos = scatter.scatterPos;

            direction = getNewDirection(seed, direction);
        }
    }

    resultRadiance.result = radiance;
    resultRadiance.importance = 0;
}

/*
* Doesn't take into account the first scattering event at all.
* Only taking into account the sun light.
*/
RT_PROGRAM void multipleScatterSunRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 radiance = make_float3(0);
    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    resultRadiance.result = make_float3(0);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y, subframeId);

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
                radiance += getInScattering(scatter, direction, true);
            }

            pos = scatter.scatterPos;

            direction = getNewDirection(seed, direction);
        }
    }

    resultRadiance.result = radiance;
    resultRadiance.importance = 0;
}