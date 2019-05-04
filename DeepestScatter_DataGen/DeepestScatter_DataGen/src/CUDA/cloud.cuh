#ifndef CLOUD_CUH
#define CLOUD_CUH

#include <optix_world.h>
#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

#include "random.cuh"
#include "optixExtraMath.cuh"

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, minimalRayDistance, , );

rtDeclareVariable(float3, bboxSize, , );
rtDeclareVariable(float3, textureScale, , );

rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float, sampleStep, , );
rtDeclareVariable(float, densityMultiplier, , );
rtDeclareVariable(float3, lightDirection, , );
rtDeclareVariable(float, lightIntensity, , );
rtDeclareVariable(float3, lightColor, , );
rtDeclareVariable(float3, skyIntensity, , );
rtDeclareVariable(float3, groundIntensity, , );

//rtTextureSampler<uchar, 3, cudaReadModeNormalizedFloat> density;
rtDeclareVariable(int, densityTextureId, , );
rtTextureSampler<uchar, 3, cudaReadModeNormalizedFloat> inScatter;
rtTextureSampler<uchar, 1, cudaReadModeNormalizedFloat> mie;
rtTextureSampler<uchar, 1, cudaReadModeNormalizedFloat> choppedMie;
rtTextureSampler<uchar, 1, cudaReadModeNormalizedFloat> choppedMieIntegral;

rtDeclareVariable(unsigned int, subframeId, , );

static __host__ __device__ __inline__ bool isInBox(float3 pos)
{
    return pos.x >= -0.01f && pos.y >= -0.01f && pos.z >= -0.01f
        && pos.x <= bboxSize.x + 0.01f && pos.y <= bboxSize.y + 0.01f && pos.z <= bboxSize.z + 0.01f;
}

static __host__ __device__ __inline__ float getMiePhase(float cosTheta)
{
    return tex1D(mie, (cosTheta + 1) / 2);
}

static __host__ __device__ __inline__ float getChoppedMiePhase(float cosTheta)
{
    return tex1D(choppedMie, (cosTheta + 1) / 2);
}

rtDeclareVariable(float3, missColor, , );

static __host__ __device__ __inline__ float sampleCloud(float3 pos)
{
    pos = pos * textureScale;
    return rtTex3D<float>(densityTextureId, pos.x, pos.y, pos.z);
}

static __host__ __device__ __inline__ float sampleInScatter(float3 pos)
{
    pos = pos * textureScale;
    return tex3D(inScatter, pos.x, pos.y, pos.z);
}

struct ScatteringEvent
{
    bool hasScattered;
    float3 scatterPos;
    float transmittance;
};

static __device__ __inline__ ScatteringEvent getNextScatteringEvent(
    unsigned int& seed,
    float3 pos, const float3& direction, bool stopAtScatterPos = true)
{
    float3 stepAlongRay = direction * sampleStep;
    float opticalDistance = rnd(seed);

    float transmittance = 1;
    bool hasScattered = false;
    float3 scatterPos = make_float3(0);

    while (isInBox(pos))
    {
        pos += stepAlongRay;

        float density = sampleCloud(pos) * densityMultiplier;
        float extinction = density * sampleStep;
        float currentTransmit = expf(-extinction);
        transmittance *= currentTransmit;

        if (!hasScattered && opticalDistance > transmittance)
        {
            hasScattered = true;
            scatterPos = pos - direction * log(opticalDistance / transmittance) / density;

            if (stopAtScatterPos)
            {
                break;
            }
        }
    }

    if (!hasScattered && !isInBox(pos))
    {
        scatterPos = pos;
    }

    return { hasScattered, scatterPos, transmittance };
}

static __device__ __inline__ const float3& sampleSky(const ScatteringEvent& scatter, const float3& direction)
{
    float3 currentLight;

    float t = clamp((direction.y + 0.5f) / 1.5f, 0.f, 1.f);
    currentLight = lerp(groundIntensity, skyIntensity, t);

    return currentLight * scatter.transmittance;
}

static __device__ __inline__ const float3& sampleSun(const float3& direction)
{
    float cosLightAngle = dot(-lightDirection, direction);

    if (cosLightAngle > 0.99998930414f) // cos(0.53 / 180 * pi / 2)
    {
        return lightColor * lightIntensity;
    }

    return make_float3(0);
}

static __device__ __inline__ const float3& getInScattering(const ScatteringEvent& scatter, const float3& direction, bool choppedMiePhase)
{
    constexpr float sunAngularRadiusDeg = 0.53f / 2;
    constexpr float sphereArea = 4 * CUDART_PI_F;
    const float sunArea = 2 * CUDART_PI_F *(1 - cos(sunAngularRadiusDeg * CUDART_PI_F / 180.0f));
    const float sunToSphereAreaRatio = sunArea / sphereArea;

    float cosLightAngle = dot(-lightDirection, direction);

    float phase = choppedMiePhase ? getChoppedMiePhase(cosLightAngle) : getMiePhase(cosLightAngle);

    return lightColor * lightIntensity * sampleInScatter(scatter.scatterPos) * phase * sunToSphereAreaRatio;
}

static __device__ __inline__ const float3& getNewDirection(unsigned int& seed, const float3& previousDirection)
{
    float l = 0.f;
    float r = 1.f;
    float m = 0.5f;

    float val = rnd(seed);
    for (int i = 0; i < 16; i++)
    {
        m = (l + r) / 2.f;
        if (val > tex1D(choppedMieIntegral, m))
        {
            l = m;
        }
        else
        {
            r = m;
        }
    }

    float cosTheta = (l + r) - 1;

    float3 newDirection = uniformOnSphereCircle(seed, cosTheta);

    Onb onb(previousDirection);
    onb.inverse_transform(newDirection);

    return normalize(newDirection);
}
#endif