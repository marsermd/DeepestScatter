//
//  voxel.cu
//  optixVolumetric
//
//  Created by Tim Tavlintsev (TVL)
//
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
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

rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> cloud;
rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> inScatter;

RT_PROGRAM void closestHitRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += make_float3(0.5f);

    float transmitance = 1;
    float luminocity = 0;
    float3 pos;
    for (float t = 0; t < 1; t += sampleStep.x)
    {
        pos = hitPoint + ray.direction * t;
        float density = tex3D(cloud, pos.x, pos.y, pos.z).x * opticalDensityMultiplier.x;
        float light = tex3D(inScatter, pos.x, pos.y, pos.z).x;

        float currentTransmit = expf(-density);

        luminocity += transmitance * (1 - currentTransmit) * light;
        transmitance *= currentTransmit;
        if (transmitance < 0.001f)
            break;
    }
    resultRadiance.result = make_float3(luminocity);
    resultRadiance.importance = transmitance;
}