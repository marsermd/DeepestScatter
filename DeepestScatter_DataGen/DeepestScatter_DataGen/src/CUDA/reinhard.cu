#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#include "random.cuh"
#include "rayData.cuh"

using namespace optix;

__device__ const float DELTA = 0.00001f;

rtBuffer<float4, 2>   progressiveBuffer;
rtBuffer<float , 1>   sumLogColumns;
rtBuffer<float , 1>   lAverage;

rtDeclareVariable(unsigned int, totalPixels, , );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtBuffer<uchar4, 2> screenBuffer;

__device__ __inline__ float getLuminance(float4 color)
{
    return dot(color, make_float4(0.265068f, 0.67023428f, 0.06409157f, 0));
}

/// For each column.
RT_PROGRAM void firstPass()
{
    size_t2 screenSize = progressiveBuffer.size();
    sumLogColumns[launchID.x] = 0;

    for (int y = 0; y < screenSize.y; y++)
    {
        float4 current = progressiveBuffer[make_uint2(launchID.x, y)];

        float luminance = getLuminance(current);

        sumLogColumns[launchID.x] += logf(luminance + DELTA);
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

rtDeclareVariable(float, exposure, , );



// First call first and second pass.
RT_PROGRAM void applyReinhard()
{
    float4 color = progressiveBuffer[launchID];
    float lw = getLuminance(color);
    float ld = lw * exposure / lAverage[0];
    ld = ld / (1.f + ld);

    float4 rgb = color * (ld / lw);

    rgb = clamp(rgb, 0.f, 1.f);

    // and gamma correction
    rgb.x = powf(rgb.x, 1.f / 2.2f);
    rgb.y = powf(rgb.y, 1.f / 2.2f);
    rgb.z = powf(rgb.z, 1.f / 2.2f);

    rgb = rgb *255;

    screenBuffer[launchID] = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
}