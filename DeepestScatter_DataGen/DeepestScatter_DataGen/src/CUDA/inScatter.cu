#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint3, launchID, rtLaunchIndex, );
rtBuffer<uchar1, 3>   resultBuffer;

rtDeclareVariable(float3, lightDirection, , );

rtDeclareVariable(float, sampleStep, , );
rtDeclareVariable(float, densityMultiplier, , );

rtDeclareVariable(float3, boxSize, , );

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
    pos = pos / boxSize;
    return tex3D(cloud, pos.x, pos.y, pos.z).x;
}

RT_PROGRAM void inScatter()
{
    size_t3 size = resultBuffer.size();
    size_t maxSize = max(max(size.x, size.y), size.z);

    float3 samplePos = make_float3(launchID) / make_float3(maxSize);

    float3 stepToLight = (-normalize(lightDirection)) * sampleStep;

    int stepCount = 1 / sampleStep;

    float transmitance = 1;
    for (int i = 0; i < stepCount; i++)
    {
        float density = sampleCloud(samplePos) * densityMultiplier;
        float extinction = density * sampleStep;

        transmitance *= expf(-extinction);
        samplePos += stepToLight;
    }
    resultBuffer[launchID] = make_uchar1(transmitance * 255);
}