#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint3, launchID, rtLaunchIndex, );
rtBuffer<uchar1, 3>   resultBuffer;

rtDeclareVariable(float3, lightDirection, , );
rtDeclareVariable(float, lightIntensity, , );

rtDeclareVariable(float, sampleStep, , );
rtDeclareVariable(float, densityMultiplier, , );

rtDeclareVariable(float3, bboxSize, , );
rtDeclareVariable(float3, textureScale, , );

rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> cloud;

inline RT_HOSTDEVICE float3 make_float3(size_t3 st)
{
    float3 ret;
    ret.x = (float)st.x;
    ret.y = (float)st.y;
    ret.z = (float)st.z;
    return ret;
}

static __host__ __device__ __inline__ float sampleCloud(float3 pos)
{
    pos = pos * textureScale;
    return tex3D(cloud, pos.x, pos.y, pos.z).x;
}

RT_PROGRAM void inScatter()
{
    size_t3 size = resultBuffer.size();
    size_t maxSize = max(max(size.x, size.y), size.z);
    float minScale = fminf(fminf(textureScale.x, textureScale.y), textureScale.z);

    float3 samplePos = (make_float3(launchID) / make_float3(maxSize)) / minScale;

    float3 stepToLight = (-normalize(lightDirection)) * sampleStep;

    int stepCount = 1 / sampleStep;

    float transmittance = 1;
    for (int i = 0; i < stepCount; i++)
    {
        float density = sampleCloud(samplePos) * densityMultiplier;
        float extinction = density * sampleStep;

        transmittance *= expf(-extinction);
        samplePos += stepToLight;
        if (transmittance * 255.f < 1.f)
        {
            break;
        }
    }
    resultBuffer[launchID] = make_uchar1(transmittance * 255.f);
}