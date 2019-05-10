#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "DisneyDescriptor.cuh"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtDeclareVariable(uint, posZ, , );
rtBuffer<LightMapNetworkInput, 2> descriptors;

RT_PROGRAM void collect()
{
    float3 origin = make_float3(launchID.x, launchID.y, posZ);
    origin /= 75;
    origin -= bboxSize * 0.5f;

    const float3 direction = make_float3(0, 0, 1);

    DisneyDescriptor descriptor;
    setupDisneyDescriptor(descriptor, origin, direction);
    descriptors[launchID].fill(descriptor);
}