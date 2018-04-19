#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#include "random.cuh"
#include "rayData.cuh"

using namespace optix;

__device__ const float DELTA = 0.001f;

rtBuffer<float4, 2>   progressiveBuffer;
rtBuffer<float , 1>   sumLogColumns;
rtBuffer<float , 1>   lAverage;

rtDeclareVariable(unsigned int, totalPixels, , );
rtDeclareVariable(uint1, launchID1, rtLaunchIndex, );


rtBuffer<uchar4, 2> screenBuffer;
/// For each column.
RT_PROGRAM void firstPass()
{
    size_t2 screenSize = progressiveBuffer.size();
    sumLogColumns[launchID1.x] = 0;

    for (int y = 0; y < screenSize.y; y++)
    {
        float4 current = progressiveBuffer[make_uint2(launchID1.x, y)];

        float luminance = dot(current, make_float4(0.2126f, 0.7152f, 0.0722f, 0));

        sumLogColumns[launchID1.x] += log(luminance + DELTA);
    }
}

/// Only one call.
RT_PROGRAM void secondPass()
{
    size_t horizontalSize = sumLogColumns.size();
    float result = 0;
    for (int i = 0; i < horizontalSize; i++)
    {
        result += sumLogColumns[i];
    }
    result = expf(result / (float)totalPixels);

    lAverage[0] = result;
}

rtDeclareVariable(float, midGrey, , );


rtDeclareVariable(uint2, launchID2, rtLaunchIndex, );

// First call first and second pass.
RT_PROGRAM void applyReinhard()
{
    float4 scaled = midGrey * progressiveBuffer[launchID2] / lAverage[0];
    scaled = scaled / (1 + scaled);

    // and gamma correction
    scaled.x = powf(scaled.x, 1.f / 2.2f);
    scaled.y = powf(scaled.y, 1.f / 2.2f);
    scaled.z = powf(scaled.z, 1.f / 2.2f);

    scaled = scaled * 255;

    screenBuffer[launchID2] = make_uchar4(scaled.x, scaled.y, scaled.z, 255);
}