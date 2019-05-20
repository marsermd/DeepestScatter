#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "DisneyDescriptor.cuh"
#include "LightProbe.h"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtDeclareVariable(uint, posZ, , );
rtBuffer<LightMapNetworkInput, 2> descriptors;

RT_PROGRAM void collect()
{
    float3 origin = make_float3(launchID.x, launchID.y, posZ) * LightProbe::STEP_IN_MEAN_FREE_PATH / densityMultiplier;
    origin -= bboxSize * 0.5f;

    const float3 direction = make_float3(0, 0, 1);

    DisneyDescriptor descriptor;
    setupHierarchicalDescriptor<DisneyDescriptor, uint8_t>(descriptor, origin, direction);
    descriptors[launchID].fill(descriptor);
}