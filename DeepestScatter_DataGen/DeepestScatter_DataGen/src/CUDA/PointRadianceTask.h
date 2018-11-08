#pragma once

#include <optixu/optixu_math_namespace.h>
#include <cinttypes>
#include <cfloat>
#include <cassert>

namespace DeepestScatter
{
    namespace Gpu
    {
        class PointRadianceTask
        {
        public:
            PointRadianceTask(int32_t id, optix::float3 position, optix::float3 direction) :
                id(id),
                position(position),
                direction(direction) {}

            /**
             * With 95% confidence
             */
            inline __device__ __host__ float getRelativeConfidenceInterval() const
            {
                return getAbsoluteConfidenceInterval() / (radiance + FLT_EPSILON);
            }

            /**
             * With 95% confidence
             */
            inline __device__ __host__ float getAbsoluteConfidenceInterval() const
            {
                const float N = experimentCount;
                const float sigma = sqrtf(runningVariance / N);
                return 1.96f * sigma / sqrtf(N);
            }

            inline __device__ __host__ void addExperimentResult(const float newRadiance)
            {
                experimentCount++;
                const float N = experimentCount;
                const float newWeight = 1.0 / N;

                const float previousMean = radiance;
                const float newMean = radiance + (newRadiance - previousMean) * newWeight;
                radiance = newMean;

                runningVariance += (newRadiance - previousMean) * (newRadiance - newMean);
            }

            /**
             * Incorporates results of other task
             */
            inline __host__ PointRadianceTask& operator+=(const PointRadianceTask& other)
            {
                assert(other.id == id);
                const float newWeight = other.experimentCount / (experimentCount + other.experimentCount);

                radiance += (other.radiance - radiance) * newWeight;
                runningVariance += other.runningVariance;
                experimentCount += other.experimentCount;

                return *this;
            }

            int32_t id;
            uint32_t experimentCount = 0;

            float radiance = 0;
            float runningVariance = 0;

            optix::float3 position;
            optix::float3 direction;
        };
    }
}
