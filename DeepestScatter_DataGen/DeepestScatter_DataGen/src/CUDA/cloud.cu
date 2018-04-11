//
//  voxel.cu
//  optixVolumetric
//
//  Created by Tim Tavlintsev (TVL)
//
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#include "random.cuh"
#include "rayData.cuh"

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, minimalRayDistance, , );


// --------------- BOX SHAPE ------------------

static __device__ void makeBox(float3 & boxmin, float3  & boxmax) {
    float halfWidth = 0.5f;
    boxmin = make_float3(-halfWidth); boxmax = make_float3(halfWidth);
}

RT_PROGRAM void intersect(int primIdx)
{
    float3 boxmin, boxmax;
    makeBox(boxmin, boxmax);

    float3 t0 = (boxmin - ray.origin) / ray.direction;
    float3 t1 = (boxmax - ray.origin) / ray.direction;
    float3 tnear = fminf(t0, t1);
    float3 tfar = fmaxf(t0, t1);
    float tmin = fmaxf(tnear);
    float tmax = fminf(tfar);

    if (tmin <= tmax) 
    {
        bool checkBack = true;
        if (rtPotentialIntersection(tmin)) 
        {
            if (rtReportIntersection(0))
            {
                checkBack = false;
            }
        }
        if (checkBack) 
        {
            if (rtPotentialIntersection(minimalRayDistance)) 
            {
                rtReportIntersection(1);
            }
        }
    }
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    float3 boxmin, boxmax;
    makeBox(boxmin, boxmax);
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->set(boxmin, boxmax);
}

// --------------- Path Tracing ------------------

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, resultRadiance, rtPayload, );

rtDeclareVariable(float1, sampleStep, , );
rtDeclareVariable(float1, opticalDensityMultiplier, , );
rtDeclareVariable(float3, lightDirection, , );

rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> cloud;
rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> inScatter;
rtTextureSampler<uchar1, 1, cudaReadModeNormalizedFloat> mie;
rtTextureSampler<uchar1, 1, cudaReadModeNormalizedFloat> choppedMie;
rtTextureSampler<uchar1, 1, cudaReadModeNormalizedFloat> choppedMieIntegral;

rtDeclareVariable(unsigned int, subframeId, rtSubframeIndex, );

static __host__ __device__ __inline__ bool isInBox(float3 pos)
{
    return pos.x >= 0.0f && pos.y >= 0.0f && pos.z >= 0.0f
        && pos.x <= 1.0f && pos.y <= 1.0f && pos.z <= 1.0f;
}

static __host__ __device__ __inline__ float getMiePhase(float cosTheta)
{
    return tex1D(mie, (cosTheta + 1) / 2).x;
}

static __host__ __device__ __inline__ float getChoppedMiePhase(float cosTheta)
{
    return tex1D(choppedMie, (cosTheta + 1) / 2).x;
}

RT_PROGRAM void closestHitRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += make_float3(0.5f);

    float3 boxmin, boxmax;
    makeBox(boxmin, boxmax);

    float3 t0 = (boxmin - ray.origin) / ray.direction;
    float3 t1 = (boxmax - ray.origin) / ray.direction;
    float3 tnear = fminf(t0, t1);
    float3 tfar = fmaxf(t0, t1);
    float tmin = fmaxf(tnear);
    float tmax = fminf(tfar);

    float radiance = 0;
    float globalTransmitance = 1;
    for (int sample = 0; sample < 20; sample++)
    {
        globalTransmitance = 1;
        float3 pos = hitPoint;

        float3 direction = normalize(ray.direction);
        int stepCount = (tmax - tmin) / sampleStep.x;

        resultRadiance.result = make_float3(0);

        unsigned int seed = tea<4>(launchID.x * 800 + launchID.y, subframeId);
        float3 normalizedLightDirection = normalize(lightDirection);

        float russianRouletteCoefficient = 1;

        int depth = 0;
        while (isInBox(pos))
        {
            float3 stepAlongRay = direction * sampleStep.x;

            float lightPhase = 1;// getChoppedMiePhase(dot(direction, normalizedLightDirection));
            float transmitance = 1;
            float currentTransmit = 1;
            while (isInBox(pos))
            {
                float extinction = tex3D(cloud, pos.x, pos.y, pos.z).x * opticalDensityMultiplier.x;

                currentTransmit = expf(-extinction);

                transmitance *= currentTransmit;

                pos += stepAlongRay;

                if (rnd(seed) > currentTransmit)
                {
                    break;
                }
            }
            globalTransmitance *= transmitance;

            if (isInBox(pos))
            {
                if (depth == 3)
                {
                    float light = tex3D(inScatter, pos.x, pos.y, pos.z).x;
                    radiance += light * russianRouletteCoefficient * getMiePhase(dot(-normalizedLightDirection, direction));
                    break;
                }
                else
                {
                    russianRouletteCoefficient *= M_PIf * 4.f;

                    float l = 0.f;
                    float r = 1.f;
                    float m = 0.5f;
                    float val = rnd(seed);
                    for (int i = 0; i < 16; i++)
                    {
                        m = (l + r) / 2.f;
                        if (tex1D(choppedMieIntegral, m).x > val)
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

                    Onb onb(direction);
                    onb.inverse_transform(newDirection);

                    direction = newDirection;
                }
            }
            depth++;
        }
    }

    resultRadiance.result = make_float3(radiance / 20);
    resultRadiance.importance = globalTransmitance;
}