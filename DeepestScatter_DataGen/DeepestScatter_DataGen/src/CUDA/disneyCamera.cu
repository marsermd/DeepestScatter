#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"
#include "random.cuh"
#include "cameraCommon.cuh"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtBuffer<DisneyNetworkInput, 2> networkInputBuffer;
rtBuffer<IntersectionInfo, 2> directRadianceBuffer;
rtBuffer<float4, 2> frameResultBuffer;

rtDeclareVariable(float3, lightDirection, , );
rtDeclareVariable(uint2, rectOrigin, , );

RT_PROGRAM void pinholeCamera()
{
    DisneyDescriptorRayData prd;
    prd.descriptor = &networkInputBuffer[launchID];
    prd.intersectionInfo = &directRadianceBuffer[launchID];

    prd.intersectionInfo->hasScattered = false;
    prd.intersectionInfo->radiance = make_float3(0);

    float3 direction = trace(prd, launchID + rectOrigin, frameResultBuffer.size());

    const float angle = acos(dot(lightDirection, direction));
    for (int i = 0; i < DisneyDescriptor::LAYERS_CNT; i++)
    {
        prd.descriptor->layers[i].angle = angle;
    }
}

RT_PROGRAM void clearRect()
{
    //todo: probably uncomment these. Right now it jut makes performance worse.
    //networkInputBuffer[launchID].clear();
    //directRadianceBuffer[launchID].radiance = make_float3(0);
    //directRadianceBuffer[launchID].hasScattered = false;
}