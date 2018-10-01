#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#define DECLARE_RAY_ID(id) static const optix::uint rayId = id;

// Dynamic initialization is not supported for a __device__ variable. So don't try initializing the variables in constructor or anywhere else.

struct RadianceRayData
{
    DECLARE_RAY_ID(0u)

    optix::float3 result;
    float importance;
};

struct ScatteringRayData
{
    DECLARE_RAY_ID(1u)

    optix::float3 position;
    bool hasScattered;
};