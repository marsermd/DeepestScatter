#pragma once
#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>

rtDeclareVariable(optix::float3, eye, , );
rtDeclareVariable(optix::float3, U, , );
rtDeclareVariable(optix::float3, V, , );
rtDeclareVariable(optix::float3, W, , );

rtDeclareVariable(rtObject, objectRoot, , );

rtDeclareVariable(float, sceneEPS, , );
rtDeclareVariable(unsigned int, subframeId, , );

// Shoot a ray in the scene and fill prd.
// Returns direction of the ray
template<typename PerRayData>
static __host__ __device__ __inline__ optix::float3 trace(PerRayData& prd, const optix::uint2& pixelId, const optix::size_t2& screenSize)
{
    using namespace optix;
    float2 d = make_float2(pixelId) / make_float2(screenSize) * 2.f - 1.f;

    float3 origin = eye;
    float3 direction = normalize(d.x*U + d.y * V + W);
    Ray ray(origin, direction, PerRayData::rayId, sceneEPS);
    rtTrace(objectRoot, ray, prd);

    return direction;
}