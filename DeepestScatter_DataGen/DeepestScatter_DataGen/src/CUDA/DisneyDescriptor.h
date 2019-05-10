#pragma once
#include <cinttypes>

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
                const static size_t LAYER_SIZE = SIZE_Z * SIZE_Y * SIZE_X;

                /**
                 * sampled from SIZE_Z × SIZE_Y × SIZE_X grid in an axis-aligned box
                 * with [−1, −1, −1] and [1, 1, 3] being two opposing corners.
                 */
                uint8_t density[LAYER_SIZE];
            };

            const static size_t LAYERS_CNT = 10;

            /**
             * Each layer's support is 2x bigger than the previous.
             */
            Layer layers[LAYERS_CNT];
        };

        class DisneyNetworkInput
        {
        public:

            __device__ __host__ DisneyNetworkInput(): layers{}
            {
            }

            __device__ __host__ inline void fill(const DisneyDescriptor& descriptor, float angle)
            {
                for (int i = 0; i < DisneyDescriptor::LAYERS_CNT; i++)
                {
                    const auto& descriptorLayer = descriptor.layers[i];
                    for (int j = 0; j < DisneyDescriptor::Layer::LAYER_SIZE; j++)
                    {
                        layers[i].density[j] = descriptorLayer.density[j] / 256.0f;
                    }
                    layers[i].angle = angle;
                }
            }

            __device__ __host__ void clear()
            {
                for (int i = 0; i < DisneyDescriptor::LAYERS_CNT; i++)
                {
                    for (int j = 0; j < DisneyDescriptor::Layer::LAYER_SIZE; j++)
                    {
                        layers[i].density[j] = 0;
                    }
                    layers[i].angle = 0;
                }
            }

            class Layer
            {
            public:

                /**
                 * sampled from SIZE_Z × SIZE_Y × SIZE_X grid in an axis-aligned box
                 * with [−1, −1, −1] and [1, 1, 3] being two opposing corners.
                 */
                float density[DisneyDescriptor::Layer::LAYER_SIZE];
                float angle;
            };

            /**
             * Each layer's support is 2x bigger than the previous.
             */
            Layer layers[DisneyDescriptor::LAYERS_CNT];
        };

        class LightMapNetworkInput
        {
        public:

            __device__ __host__ LightMapNetworkInput() : layers{}
            {
            }

            __device__ __host__ inline void fill(const DisneyDescriptor& descriptor)
            {
                for (int i = 0; i < DisneyDescriptor::LAYERS_CNT; i++)
                {
                    const auto& descriptorLayer = descriptor.layers[i];
                    for (int j = 0; j < DisneyDescriptor::Layer::LAYER_SIZE; j++)
                    {
                        layers[i].density[j] = descriptorLayer.density[j] / 256.0f;
                    }
                }
            }

            class Layer
            {
            public:

                /**
                 * sampled from SIZE_Z × SIZE_Y × SIZE_X grid in an axis-aligned box
                 * with [−1, −1, −1] and [1, 1, 3] being two opposing corners.
                 */
                float density[DisneyDescriptor::Layer::LAYER_SIZE];
            };

            /**
             * Each layer's support is 2x bigger than the previous.
             */
            Layer layers[DisneyDescriptor::LAYERS_CNT];
        };
    }
}