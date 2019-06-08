#pragma once

#include "LightProbe.h"
#include <optix_device.h>
#include "DisneyDescriptor.cuh"


using namespace optix;
using namespace DeepestScatter::Gpu;
rtBuffer<LightProbe, 3> bakedLightProbes;

static __host__ __device__ __inline__ optix::uint3 floorId(float3 v, float3& remainder)
{
    v = v * densityMultiplier / LightProbe::STEP_IN_MEAN_FREE_PATH;
    const int3 id = optix::make_int3(
        static_cast<int32_t>(v.x),
        static_cast<int32_t>(v.y),
        static_cast<int32_t>(v.z)
    );
    remainder = v - make_float3(id);
    const int3 res = max(id, make_int3(0));
    return make_uint3(res.x, res.y, res.z);
}

static __host__ __device__ __inline__ float sqrLength(const float3& a)
{
    return
        a.x * a.x +
        a.y * a.y +
        a.z * a.z;
}

static __host__ __device__ __inline__ float length(float x, float y, float z)
{
    return sqrtf
    (
        x * x +
        y * y +
        z * z
    );
}

static __host__ __device__ __inline__ void lerp(const LightProbe& a, const LightProbe& b, float t, LightProbe& result)
{
    for (int i = 0; i < result.LENGTH; i++)
    {
        result.data[i] = lerp(a.data[i], b.data[i], t);
    }
}

//scalar tripple product
static __host__ __device__ __inline__ float ScTP(const float3& a, const float3& b, const float3& c)
{
    return dot(cross(a, b), c);
}

static __host__ __device__ __inline__ float4 barycentric(const float3& a, const float3& b, const float3& c, const float3& d, const float3& p)
{
    float3 vap = p - a;
    float3 vbp = p - b;

    float3 vab = b - a;
    float3 vac = c - a;
    float3 vad = d - a;

    float3 vbc = c - b;
    float3 vbd = d - b;

    float va6 = ScTP(vbp, vbd, vbc);
    float vb6 = ScTP(vap, vac, vad);
    float vc6 = ScTP(vap, vad, vab);
    float vd6 = ScTP(vap, vab, vac);
    float v6 = 1 / ScTP(vab, vac, vad);
    return make_float4(va6*v6, vb6*v6, vc6*v6, vd6*v6);
}

class LightProbeInterpolation
{
public:
    __host__ __device__ __inline__ LightProbeInterpolation(
        const uint3& id, const float3& local,
        const uint3& a,
        const uint3& b,
        const uint3& c,
        const uint3& d) :
        a(id + a),
        b(id + b),
        c(id + c),
        d(id + d)
    {
        // a, b, c, d from the parameters!
        w = barycentric
        (
            make_float3(a),
            make_float3(b),
            make_float3(c),
            make_float3(d),
            local
        );
    }

    // Light Probe Positions
    uint3 a, b, c, d;

    // Light Probe Weights
    float4 w;
};

static __host__ __device__ __inline__ void lerp(
    const LightProbeInterpolation& interpolation,
    LightProbeRendererInput::Probe& result)
{
    uint8_t* la = bakedLightProbes[interpolation.a].data;
    uint8_t* lb = bakedLightProbes[interpolation.b].data;
    uint8_t* lc = bakedLightProbes[interpolation.c].data;
    uint8_t* ld = bakedLightProbes[interpolation.d].data;

    for (size_t i = 0; i < LightProbe::LENGTH; i++)
    {
        result.data[i] =
            (
                interpolation.w.x * la[i] +
                interpolation.w.y * lb[i] +
                interpolation.w.z * lc[i] +
                interpolation.w.w * ld[i]
            ) / 256.0f;
    }
}

static __host__ __device__ __inline__ bool isCloseToVertex(const float3& localOffset, const float3& vertex)
{
    constexpr float distanceToPlane = 0.577350269190; // 1 / sqrt(3)
    const float3 normal = normalize(make_float3(0.5f) - vertex);
    return dot(localOffset - vertex, normal) < distanceToPlane;
}

static __host__ __device__ __inline__ LightProbeInterpolation getLightProbeInterpolation(const float3& v)
{
    float3 localOffset;
    uint3 id = floorId(v, localOffset);

    uint3 a, b, c, d;
    if (isCloseToVertex(localOffset, make_float3(0, 0, 0)))
    {
        a = make_uint3(0);
        b = make_uint3(1, 0, 0);
        c = make_uint3(0, 1, 0);
        d = make_uint3(0, 0, 1);
    }
    else if (isCloseToVertex(localOffset, make_float3(0, 1, 1)))
    {
        a = make_uint3(0, 1, 1);
        b = make_uint3(0, 0, 1);
        c = make_uint3(0, 1, 0);
        d = make_uint3(1, 1, 1);
    }
    else if (isCloseToVertex(localOffset, make_float3(1, 0, 1)))
    {
        a = make_uint3(1, 0, 1);
        b = make_uint3(0, 0, 1);
        c = make_uint3(1, 0, 0);
        d = make_uint3(1, 1, 1);
    }
    else if (isCloseToVertex(localOffset, make_float3(1, 1, 0)))
    {
        a = make_uint3(1, 1, 0);
        b = make_uint3(0, 1, 0);
        c = make_uint3(1, 0, 0);
        d = make_uint3(1, 1, 1);
    }
    else
    {
        a = make_uint3(1, 0, 0);
        b = make_uint3(0, 1, 0);
        c = make_uint3(0, 0, 1);
        d = make_uint3(1, 1, 1);
    }
    
    return LightProbeInterpolation(id, localOffset, a, b, c, d);
}


static __host__ __device__ __inline__ void interpolateLightProbe(const float3& v, LightProbeRendererInput::Probe& result)
{
    LightProbeInterpolation interpolation = getLightProbeInterpolation(v);
    lerp(interpolation, result);

    //Trilinear interpolation:
    //LightProbe c00;
    //lerp(select(id0, id1, 0, 0, 0), select(id0, id1, 1, 0, 0), local.x, c00);
    //LightProbe c10;
    //lerp(select(id0, id1, 0, 1, 0), select(id0, id1, 1, 1, 0), local.x, c10);
    //LightProbe c0;
    //lerp(c00, c10, local.y, c0);

    //LightProbe& c01 = c00;
    //lerp(select(id0, id1, 0, 0, 1), select(id0, id1, 1, 0, 1), local.x, c01);
    //LightProbe& c11 = c10;
    //lerp(select(id0, id1, 0, 1, 1), select(id0, id1, 1, 1, 1), local.x, c11);
    //LightProbe c1;
    //lerp(c01, c11, local.y, c1);
    //lerp(c0, c1, local.z, result);
}