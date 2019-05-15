#pragma once
#include <optixu/optixu_math_namespace.h>

namespace DeepestScatter
{
    namespace Gpu
    {
        class LightProbe
        {
        public:
            static const size_t RESOLUTION = 150;
            static const size_t PROBE_COUNT = RESOLUTION + 1;

            static const size_t LENGTH = 200;
            float data[LENGTH];
        };

        class LightProbeRendererInput
        {
        public:
            LightProbe lightProbe;

            float omega;
            float alpha;
        };
    }
}
