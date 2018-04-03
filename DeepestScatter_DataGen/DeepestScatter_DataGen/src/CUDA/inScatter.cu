#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint3, launchID, rtLaunchIndex, );
rtBuffer<float1, 3>   resultBuffer;

rtDeclareVariable(float3, lightDirection, , );
rtDeclareVariable(float1, lightIntensity, , );

rtDeclareVariable(float1, sampleStep, , );
rtDeclareVariable(float1, opticalDensityMultiplier, , );

rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> cloud;

inline RT_HOSTDEVICE float3 make_float3(size_t3 st)
{
    float3 ret;
    ret.x = (float)st.x;
    ret.y = (float)st.y;
    ret.z = (float)st.z;
    return ret;
}

RT_PROGRAM void inScatter()
{
    size_t3 size = resultBuffer.size();

    float3 samplePos = make_float3(launchID) / make_float3(size);

    float3 stepToLight = (-normalize(lightDirection)) * sampleStep.x;

    int stepCount = 1 / sampleStep.x;

    float transmitance = 1;
    for (int i = 0; i < stepCount; i++)
    {
        // TODO:division by 2 is an temporary hack to produce better images until multiple scatter is implemented
        float density = tex3D(cloud, samplePos.x, samplePos.y, samplePos.z).x * opticalDensityMultiplier.x / 6;

        transmitance *= expf(-density);
        samplePos += stepToLight;
    }
    resultBuffer[launchID] = make_float1(lightIntensity.x * transmitance);
}