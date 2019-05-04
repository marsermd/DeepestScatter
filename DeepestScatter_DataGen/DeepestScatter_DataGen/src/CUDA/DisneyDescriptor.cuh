#pragma once

#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>

#include <gsl/gsl>
#include "optixExtraMath.cuh"
#include "DisneyDescriptor.h"

using namespace optix;

#ifndef CLOUD_CUH
rtDeclareVariable(float3, lightDirection, , );

rtDeclareVariable(float, densityMultiplier, , );

rtDeclareVariable(float3, bboxSize, , );
rtDeclareVariable(float3, textureScale, , );

rtDeclareVariable(int, densityTextureId, , );
#endif

rtDeclareVariable(float, voxelSizeInTermsOfFreePath, , );

inline RT_HOSTDEVICE float3 make_float3(size_t3 st)
{
    float3 ret;
    ret.x = gsl::narrow_cast<float>(st.x);
    ret.y = gsl::narrow_cast<float>(st.y);
    ret.z = gsl::narrow_cast<float>(st.z);
    return ret;
}


static __host__ __device__ __inline__ float sampleCloud(float3 pos, float mipLevel)
{
    pos = pos * textureScale;
    return rtTex3DLod<float>(densityTextureId, pos.x, pos.y, pos.z, fmaxf(0.0f, mipLevel));
}

namespace DeepestScatter
{
    namespace Gpu
    {
        __device__ __inline__ void setupDisneyDescriptor(DisneyDescriptor& descriptor, float3 worldPos, float3 viewDirection)
        {
            const float3 eZ = normalize(lightDirection);
            const float3 eX = normalize(cross(lightDirection, viewDirection));
            const float3 eY = cross(eX, eZ);

            const float3 origin = worldPos + 0.5f * bboxSize;

            // 0.5f so that [−1, −1, −1] and [1, 1, 3] are in two opposing corners
            float scale = 0.5f / densityMultiplier;
            // -1 because there are two sample points between 0 and 1.
            float mipmapLevel = -log2f(voxelSizeInTermsOfFreePath) - 1;

            for (size_t layerId = 0; layerId < DisneyDescriptor::LAYERS_CNT; layerId++)
            {
                DisneyDescriptor::Layer& layer = descriptor.layers[layerId];

                uint32_t sampleId = 0;
                for (int z = -2; z <= 6; z++)
                {
                    for (int y = -2; y <= 2; y++)
                    {
                        for (int x = -2; x <= 2; x++)
                        {
                            float3 offset = (eX * x + eY * y + eZ * z) * scale;
                            const float3 pos = origin + offset;
                            layer.density[sampleId] = make_uchar1(sampleCloud(pos, mipmapLevel) * 255.0f).x;
                            sampleId++;
                        }
                    }
                }

                scale *= 2;
                mipmapLevel++;
            }
        }
    }
}