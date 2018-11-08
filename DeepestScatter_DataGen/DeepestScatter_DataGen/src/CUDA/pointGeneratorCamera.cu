#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"
#include "random.cuh"
#include <cassert>
#include <gsl/gsl_util>

using namespace optix;

rtDeclareVariable(uint, launchID, rtLaunchIndex, );
rtBuffer<float3, 1> directionBuffer;
rtBuffer<float3, 1> positionBuffer;

rtDeclareVariable(rtObject, objectRoot, , );
rtDeclareVariable(float3, errorColor, , );

rtDeclareVariable(float, sceneEPS, , );

RT_PROGRAM void generatePoints()
{
    unsigned int seed = tea<6>(launchID);
    while (true)
    {
        float3 discNormal = uniformOnSphere(seed);
        float discRadius = sqrtf(3) / 2;
        float3 position = uniformOnDisc(seed, discNormal) * discRadius;

        ScatteringRayData scatter;
        scatter.position = make_float3(NAN);
        scatter.hasScattered = false;
        optix::Ray ray(position + discNormal * 2, -discNormal, scatter.rayId, sceneEPS);
        rtTrace(objectRoot, ray, scatter);

        if (scatter.hasScattered)
        {
            positionBuffer[launchID] = scatter.position;
            directionBuffer[launchID] = -discNormal;
            break;
        }
    }
}

RT_PROGRAM void clear()
{
    positionBuffer[launchID] = make_float3(NAN);
    directionBuffer[launchID] = make_float3(NAN);
}

RT_PROGRAM void exception()
{
    // We would rather want to beeak the program during the data generation than continue with an error.
    assert(0);
}

RT_PROGRAM void miss()
{
    // We don't want to sample sky or anything except for the cloud.
    assert(0);
}