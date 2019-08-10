#pragma once

#include <optixu/optixu_math_namespace.h>
#include "DisneyDescriptor.h"
#include "LightProbe.h"

#define DECLARE_RAY_ID(id) static const optix::uint rayId = id;

// WARNING! Dynamic initialization is not supported for a __device__ variable. 
// So don't try initializing member variables in constructor or anywhere else.

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

struct IntersectionInfo
{
    optix::float3 radiance;
    float transmittance;
    bool hasScattered;
};

struct DisneyDescriptorRayData
{
    DECLARE_RAY_ID(2u)

    DeepestScatter::Gpu::DisneyNetworkInput* descriptor;
    IntersectionInfo* intersectionInfo;
};

struct LightProbeRayData
{
    DECLARE_RAY_ID(3u)

    DeepestScatter::Gpu::LightProbeRendererInput* lightProbe;
    DeepestScatter::Gpu::BakedRendererDescriptor* descriptor;
    IntersectionInfo* intersectionInfo;
};