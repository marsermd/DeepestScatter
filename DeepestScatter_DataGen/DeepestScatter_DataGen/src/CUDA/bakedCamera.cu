#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "rayData.cuh"
#include "LightProbe.h"

#include "cameraCommon.cuh"

using namespace optix;
using namespace DeepestScatter::Gpu;

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );
rtBuffer<LightProbeRendererInput, 2> lightProbeInputBuffer;
rtBuffer<BakedRendererDescriptor, 2> descriptorInputBuffer;
rtBuffer<IntersectionInfo, 2> directRadianceBuffer;
rtBuffer<float, 2> predictedRadianceBuffer;
rtBuffer<float4, 2> frameResultBuffer;

rtDeclareVariable(float3, lightDirection, , );

rtDeclareVariable(uint2, rectOrigin, , );

RT_PROGRAM void pinholeCamera()
{
    LightProbeRayData prd;
    prd.intersectionInfo = &directRadianceBuffer[launchID];
    prd.lightProbe = &lightProbeInputBuffer[launchID];
    prd.descriptor = &descriptorInputBuffer[launchID];

    prd.intersectionInfo->hasScattered = false;
    prd.intersectionInfo->radiance = make_float3(0);

    trace(prd, launchID + rectOrigin, frameResultBuffer.size());
}

RT_PROGRAM void copyToFrameResult()
{
    if (directRadianceBuffer[launchID].hasScattered)
    {
        frameResultBuffer[launchID + rectOrigin] = 
            (make_float4(predictedRadianceBuffer[launchID]) +
            make_float4(directRadianceBuffer[launchID].radiance)) * (1 - directRadianceBuffer[launchID].transmittance);
    }
}