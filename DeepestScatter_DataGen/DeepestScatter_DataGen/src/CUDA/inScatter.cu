#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <gsl/gsl>

#include "optixExtraMath.cuh"

using namespace optix;

rtDeclareVariable(uint3, launchID, rtLaunchIndex, );
rtBuffer<uchar1, 3>   inScatterBuffer;

rtDeclareVariable(float3, lightDirection, , );
rtDeclareVariable(float, lightIntensity, , );

rtDeclareVariable(float, sampleStep, , );
rtDeclareVariable(float, densityMultiplier, , );

rtDeclareVariable(float3, bboxSize, , );
rtDeclareVariable(float3, textureScale, , );

rtTextureSampler<uchar, 3, cudaReadModeNormalizedFloat> density;

inline RT_HOSTDEVICE float3 make_float3(size_t3 st)
{
    float3 ret;
    ret.x = gsl::narrow_cast<float>(st.x);
    ret.y = gsl::narrow_cast<float>(st.y);
    ret.z = gsl::narrow_cast<float>(st.z);
    return ret;
}


static __host__ __device__ __inline__ float sampleCloud(float3 pos)
{
    pos = pos * textureScale;
    return tex3D(density, pos.x, pos.y, pos.z) * 2;
}

RT_PROGRAM void inScatter()
{
    const size_t3 size = inScatterBuffer.size();
    const size_t maxSize = max(max(size.x, size.y), size.z);
    const float minScale = fminf(fminf(textureScale.x, textureScale.y), textureScale.z);

    float3 samplePos = (make_float3(launchID) / make_float3(maxSize)) / minScale;

    const float3 stepToLight = (-normalize(lightDirection)) * sampleStep;

    const int stepCount = 1 / sampleStep;

    float transmittance = 1;
    for (int i = 0; i < stepCount; i++)
    {
        const float density = sampleCloud(samplePos) * densityMultiplier;
        const float extinction = density * sampleStep;

        transmittance *= expf(-extinction);
        samplePos += stepToLight;
        if (transmittance * 255.f < 1.f)
        {
            break;
        }
    }
    inScatterBuffer[launchID] = make_uchar1(transmittance * 255.f);
}