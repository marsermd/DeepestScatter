#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "DisneyDescriptor.cuh"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint, launchID, rtLaunchIndex, );

rtBuffer<DisneyDescriptor, 1> descriptors;

rtBuffer<float3, 1> directionBuffer;
rtBuffer<float3, 1> positionBuffer;

rtDeclareVariable(rtObject, objectRoot, , );

rtDeclareVariable(float, sceneEPS, , );
rtDeclareVariable(unsigned int, subframeId, , );


RT_PROGRAM void collect()
{
    const float3 origin = positionBuffer[launchID];
    const float3 direction = directionBuffer[launchID];

    DisneyDescriptor& descriptor = descriptors[launchID];
    setupDisneyDescriptor(descriptor, origin, direction);
}

RT_PROGRAM void clear()
{
    constexpr size_t pointsInLayer =
        DisneyDescriptor::Layer::SIZE_Z *
        DisneyDescriptor::Layer::SIZE_Y *
        DisneyDescriptor::Layer::SIZE_X;

    DisneyDescriptor& descriptor = descriptors[launchID];

    for (size_t layer = 0; layer < DisneyDescriptor::LAYERS_CNT; layer++)
    {
        for (size_t id = 0; id < pointsInLayer; id++)
        {
            descriptor.layers[layer].density[id] = 0;
        }
    }
}