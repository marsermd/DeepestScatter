#pragma once
#include <cinttypes>
#include <gsl/gsl>

namespace DeepestScatter
{
    namespace Gpu
    {
        class DisneyDescriptor
        {
        public:
            class Layer
            {
            public:
                const static size_t SIZE_X = 5;
                const static size_t SIZE_Y = 5;
                const static size_t SIZE_Z = 9;

                /**
                 * sampled from SIZE_Z × SIZE_Y × SIZE_X grid in an axis-aligned box
                 * with [−1, −1, −1] and [1, 1, 3] being two opposing corners.
                 */
                uint8_t density[SIZE_Z * SIZE_Y * SIZE_X];
            };

            const static size_t LAYERS_CNT = 10;

            /**
             * Each layer's support is 2x bigger than the previous.
             */
            Layer layers[LAYERS_CNT];
        };
    }
}