
#include "cloud.cuh"
#include "LightProbe.h"
#include "LightProbe.cuh"

#include "rayData.cuh"
#include <optix_device.h>
#include "DisneyDescriptor.cuh"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(LightProbeRayData, rayData, rtPayload, );
rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

static __host__ __device__ __inline__ float getSignedAngle(const float3& v1, const float3& v2, const float3& normal)
{
    float angle = acos(dot(v1, v2));

    if (dot(normal, cross(v1, v2)) < 0)
    {
        angle = -angle;
    }
    return angle;
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

    const float transmittance = getNextScatteringEvent(seed, pos, direction, false).transmittance;
    const ScatteringEvent scatter = getNextScatteringEvent(1 - rnd(seed) * (1 - transmittance), pos, direction);

    rayData.intersectionInfo->transmittance = transmittance;

    if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
    {
        rayData.intersectionInfo->hasScattered = false;
        rayData.intersectionInfo->radiance = make_float3(0);
    }
    else
    {
        rayData.intersectionInfo->hasScattered = true;
        rayData.intersectionInfo->radiance = getInScattering(scatter, direction, false);

        //TODO: REMOVE
        // auto interpolationData = getLightProbeInterpolation(scatter.scatterPos);
        // rayData.intersectionInfo->radiance =
        //     make_float3(bakedLightProbes[interpolationData.a].data[1], 0, 0) * interpolationData.w.x +
        //     make_float3(bakedLightProbes[interpolationData.b].data[1], 0, 0) * interpolationData.w.y +
        //     make_float3(bakedLightProbes[interpolationData.c].data[1], 0, 0) * interpolationData.w.z +
        //     make_float3(bakedLightProbes[interpolationData.d].data[1], 0, 0) * interpolationData.w.w;
        // rayData.intersectionInfo->radiance /= 5;
        
        const float3 eZ1 = normalize(lightDirection);
        const float3 eX1 = normalize(cross(lightDirection, direction));
        const float3 eY1 = cross(eX1, eZ1);

        const float3 eZ2 = normalize(lightDirection);
        const float3 eX2 = normalize(cross(lightDirection, make_float3(0, 0, 1)));
        const float3 eY2 = cross(eX2, eZ2);

        rayData.lightProbe->omega = acos(dot(lightDirection, direction));
        rayData.lightProbe->alpha = getSignedAngle(eY1, eY2, eZ1);
        
        interpolateLightProbe(scatter.scatterPos, rayData.lightProbe->probe);
        
        setupHierarchicalDescriptor<BakedRendererDescriptor::Descriptor, float>(rayData.descriptor->descriptor, scatter.scatterPos - 0.5f * bboxSize, direction);
        
        for (size_t layer = 0; layer < rayData.descriptor->descriptor.LAYERS_CNT; layer++)
        {
            rayData.descriptor->descriptor.layers[layer].meta.omega = rayData.lightProbe->omega;
        }
    }

}