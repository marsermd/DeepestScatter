#include "cloud.cuh"
#include "rayData.cuh"

static const int MAX_DEPTH = 2000;

rtDeclareVariable(RadianceRayData, resultRadiance, rtPayload, );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

RT_PROGRAM void totalRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 radiance = make_float3(0);
    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    resultRadiance.result = make_float3(0);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    float skySampleProbability = 0.1f;
    //Uncomment to enable sky:
    bool shouldSampleSky = false;//subframeId % 10 == 0;

    int depth = 0;
    while (isInBox(pos))
    {
        depth++;
        if (depth == MAX_DEPTH)
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

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    direction = getNewDirection(seed, direction);
    int depth = 0;
    while (isInBox(pos))
    {
        depth++;
        if (depth == MAX_DEPTH)
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
            // next event estimation
            radiance += getInScattering(scatter, direction, true);

            pos = scatter.scatterPos;

            direction = getNewDirection(seed, direction);
        }
    }

    resultRadiance.result = radiance;
    resultRadiance.importance = 0;
}

/*
* Only the direct light of the sun.
*/
RT_PROGRAM void singleScatterSunRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 radiance = make_float3(0);
    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    resultRadiance.result = make_float3(0);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    ScatteringEvent scatter = getNextScatteringEvent(seed, pos, direction);

    if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
    {
        // do nothing
    }
    else
    {
        // next event estimation
        radiance += getInScattering(scatter, direction, false);
    }

    resultRadiance.result = radiance;
    resultRadiance.importance = 0;
}