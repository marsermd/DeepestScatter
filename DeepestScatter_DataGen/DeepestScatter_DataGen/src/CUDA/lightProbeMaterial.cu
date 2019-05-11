/*
* Only the direct light of the sun.
*/
#include "cloud.cuh"
#include "rayData.cuh"
#include <optix_device.h>
#include "LightProbe.h"
#include "DisneyDescriptor.cuh"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(LightProbeRayData, rayData, rtPayload, );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtBuffer<LightProbe, 3> bakedLightProbes;

static __host__ __device__ __inline__ optix::uint3 floorId(float3 v, float step, float3& remainder)
{
    v *= step;
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
    // ScTP computes the scalar triple product
    float va6 = ScTP(vbp, vbd, vbc);
    float vb6 = ScTP(vap, vac, vad);
    float vc6 = ScTP(vap, vad, vab);
    float vd6 = ScTP(vap, vab, vac);
    float v6 = 1 / ScTP(vab, vac, vad);
    return make_float4(va6*v6, vb6*v6, vc6*v6, vd6*v6);
}

static __host__ __device__ __inline__ void lerp(
    const uint3& id, const float3& local,
    const uint3& a,
    const uint3& b,
    const uint3& c,
    const uint3& d,
    LightProbe& result)
{
    float4 w = barycentric(
        make_float3(a),
        make_float3(b), 
        make_float3(c), 
        make_float3(d), 
        local);

    float* la = bakedLightProbes[id + a].data;
    float* lb = bakedLightProbes[id + b].data;
    float* lc = bakedLightProbes[id + c].data;
    float* ld = bakedLightProbes[id + d].data;

    //uint3 debug[] = {
    //    id + a,
    //    id + b,
    //    id + c,
    //    id + d
    //};
    //for (size_t i = 0; i < 16; i++)
    //{
    //    if (reinterpret_cast<uint32_t*>(debug)[i] > 35)
    //    {
    //        rtPrintf("lol %u", reinterpret_cast<uint32_t*>(debug)[i]);
    //    }
    //}

    for (size_t i = 0; i < result.LENGTH; i++)
    {
        result.data[i] =
            w.x * la[i] +
            w.y * lb[i] +
            w.z * lc[i] +
            w.w * ld[i];
    }
}

static __host__ __device__ __inline__ void interpolateLightProbe(LightProbe& result, const float3& v)
{
    constexpr float distanceToPlane = 1 / 0.577350269190; // 1 / sqrt(3)
    constexpr float sqrDistanceToPlane = 1 / 3;

    float3 local;
    uint3 id = floorId(v, 75, local);

    uint3 a;
    uint3 b;
    uint3 c;
    uint3 d;

    if (sqrLength(local) < sqrDistanceToPlane)
    {
        a = make_uint3(0);
        b = make_uint3(1, 0, 0);
        c = make_uint3(0, 1, 0);
        d = make_uint3(0, 0, 1);
    }
    else if (sqrLength(local - make_float3(0, 1, 1)) < sqrDistanceToPlane)
    {
        a = make_uint3(0, 1, 1);
        b = make_uint3(0, 0, 1);
        c = make_uint3(0, 1, 0);
        d = make_uint3(1, 1, 1);
    }
    else if (sqrLength(local - make_float3(1, 0, 1)) < sqrDistanceToPlane)
    {
        a = make_uint3(1, 0, 1);
        b = make_uint3(0, 0, 1);
        c = make_uint3(1, 0, 0);
        d = make_uint3(1, 1, 1);
    }
    else if (sqrLength(local - make_float3(1, 1, 0)) < sqrDistanceToPlane)
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

    lerp(id, local, a, b, c, d, result);

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

RT_PROGRAM void sampleLightProbe()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * bboxSize;

    float3 pos = hitPoint;

    const float3 direction = normalize(ray.direction);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y);

    //const uint step = 50;
    //const float jitter = rnd(seed) / step;
    //float opticalDistance = (subframeId % step) * 1.0f / step + jitter;

    const ScatteringEvent scatter = getNextScatteringEvent(seed, pos, direction);

    if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
    {
        rayData.intersectionInfo->hasScattered = false;
    }
    else
    {
        rayData.intersectionInfo->hasScattered = true;
        rayData.intersectionInfo->radiance = getInScattering(scatter, direction, false);

        const float3 eZ1 = normalize(lightDirection);
        const float3 eX1 = normalize(cross(lightDirection, direction));
        const float3 eY1 = cross(eX1, eZ1);

        const float3 eZ2 = normalize(lightDirection);
        const float3 eX2 = normalize(cross(lightDirection, make_float3(0, 0, 1)));
        const float3 eY2 = cross(eX2, eZ2);

        rayData.lightProbe->omega = acos(dot(lightDirection, direction));
        rayData.lightProbe->alpha = acos(dot(eY1, eY2));

        interpolateLightProbe(rayData.lightProbe->lightProbe, scatter.scatterPos);

        setupHierarchicalDescriptor<BakedRendererDescriptor::Descriptor, float> (rayData.descriptor->descriptor, scatter.scatterPos - 0.5f * bboxSize, direction);
        for (size_t layer = 0; layer < rayData.descriptor->descriptor.LAYERS_CNT; layer++)
        {
            rayData.descriptor->descriptor.layers[layer].meta.omega = rayData.lightProbe->omega;
            rayData.descriptor->descriptor.layers[layer].meta.alpha = rayData.lightProbe->alpha;
        }
    }

}