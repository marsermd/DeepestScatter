#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "DisneyDescriptor.cuh"
#include "LightProbe.cuh"
#include "LightProbe.h"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint, launchID, rtLaunchIndex, );

rtBuffer<BakedInterpolationSet, 1> interpolationSets;

rtBuffer<float3, 1> directionBuffer;
rtBuffer<float3, 1> positionBuffer;

rtDeclareVariable(rtObject, objectRoot, , );

rtDeclareVariable(float, sceneEPS, , );
rtDeclareVariable(unsigned int, subframeId, , );

static __host__ __device__ __inline__ float3 probeIdToWorld(size_t3 id)
{
    return make_float3(id) * LightProbe::STEP_IN_MEAN_FREE_PATH / densityMultiplier - bboxSize * 0.5f;
}

static __host__ __device__ __inline__ float3 setupInterpolationProbe(
    BakedInterpolationSet::Probe& probe, 
    const size_t3& id, float power, const float3& direction)
{
    const float3 pos = probeIdToWorld(id);

    //TODO:
    setupHierarchicalDescriptor<DisneyDescriptor, uint8_t>(probe.descriptor, pos, direction);
    probe.power = power;
    probe.position = pos;
    probe.direction = direction;
}

RT_PROGRAM void collect()
{
    const float3 origin = positionBuffer[launchID] + bboxSize * 0.5f;
    const float3 direction = directionBuffer[launchID];

    const LightProbeInterpolation interpolationSettings = getLightProbeInterpolation(origin);

    BakedInterpolationSet& interpolationResult = interpolationSets[launchID];
    setupInterpolationProbe(interpolationResult.a, interpolationSettings.a, interpolationSettings.w.x, direction);
    setupInterpolationProbe(interpolationResult.b, interpolationSettings.b, interpolationSettings.w.y, direction);
    setupInterpolationProbe(interpolationResult.c, interpolationSettings.c, interpolationSettings.w.z, direction);
    setupInterpolationProbe(interpolationResult.d, interpolationSettings.d, interpolationSettings.w.w, direction);
}

RT_PROGRAM void clear()
{
    BakedInterpolationSet& interpolationSet = interpolationSets[launchID];
    std::memset(&interpolationSet, 0, sizeof(BakedInterpolationSet));
}