#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

struct PerRayData_radiance
{
    optix::float3 result;
    float importance;
    int depth;
};
