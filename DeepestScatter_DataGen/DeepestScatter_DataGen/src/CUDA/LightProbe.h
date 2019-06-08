#pragma once
#include "DisneyDescriptor.h"

namespace DeepestScatter
{
    namespace Gpu
    {
        class LightProbe
        {
        public:
            static const size_t STEP_IN_MEAN_FREE_PATH = 6;

            static const size_t LENGTH = 200;
            uint8_t data[LENGTH];
        };

        class LightProbeRendererInput
        {
        public:
            class Probe
            {
            public:
                float data[LightProbe::LENGTH];
            };

            Probe probe;
            float omega;
            float alpha;
        };

        class BakedInterpolationSet
        {
        public:
            class Probe
            {
            public:
                DisneyDescriptor descriptor;
                float power;
                optix::float3 position;
                optix::float3 direction;
            };

            Probe a;
            Probe b;
            Probe c;
            Probe d;
        };
    }
}
