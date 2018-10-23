#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

using namespace optix;

__device__ const float DELTA = 0.00001f;

rtBuffer<float4, 2>   progressiveBuffer;
rtBuffer<float4, 2>   varianceBuffer;
rtBuffer<float , 1>   sumLuminanceColumns;
rtBuffer<float , 1>   averageLuminance;

rtDeclareVariable(unsigned int, totalPixels, , );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtDeclareVariable(unsigned int, subframeId, , );

rtBuffer<uchar4, 2> screenBuffer;

__device__ __inline__ float getLuminance(float4 color)
{
    return dot(color, make_float4(0.265068f, 0.67023428f, 0.06409157f, 0));
}

/// Call for each column.
RT_PROGRAM void firstPass()
{
    size_t2 screenSize = progressiveBuffer.size();
    sumLuminanceColumns[launchID.x] = 0;

    for (int y = 0; y < screenSize.y; y++)
    {
        float4 current = progressiveBuffer[make_uint2(launchID.x, y)];

        float luminance = getLuminance(current);

        // Normally we would calculate a log-average, but here we are trying to display the brightest part in it's best (the cloud)
        // And therefore we will use a simple average.
        sumLuminanceColumns[launchID.x] += luminance + DELTA;
    }
}

/// Call only once for the whole image
RT_PROGRAM void secondPass()
{
    size_t horizontalSize = sumLuminanceColumns.size();
    float result = 0;
    for (int i = 0; i < horizontalSize; i++)
    {
        result += sumLuminanceColumns[i];
    }
    result = result / (float)totalPixels;

    averageLuminance[0] = result;
}

rtDeclareVariable(float, exposure, , );



// Call for each pixel of the image after previous two passes
RT_PROGRAM void applyReinhard()
{
    float4 color = progressiveBuffer[launchID];
    float lw = getLuminance(color);
    float ld = lw * exposure / averageLuminance[0];
    ld = ld / (1.f + ld);

    float4 rgb = color * (ld / lw);

    rgb = clamp(rgb, 0.f, 1.f);

    // and gamma correction
    rgb.x = powf(rgb.x, 1.f / 2.2f);
    rgb.y = powf(rgb.y, 1.f / 2.2f);
    rgb.z = powf(rgb.z, 1.f / 2.2f);


    //float N = subframeId;
    //rgb = make_float4(clamp(sqrt(varianceBuffer[launchID].z / N) / progressiveBuffer[launchID].z / sqrtf(N), 0.f, 1.f));
    rgb = rgb * 255;

    screenBuffer[launchID] = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
}