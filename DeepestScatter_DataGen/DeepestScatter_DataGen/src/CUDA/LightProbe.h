#pragma once
#include <optixu/optixu_math_namespace.h>

namespace DeepestScatter
{
    namespace Gpu
    {
        class LightProbe
        {
        public:
            float data[200];
        };

        class LightProbeRendererInput
        {
        public:
            LightProbe lightProbe;

            float omega;
            float alpha;
            optix::float3 offset;
        };
    }
}
