#pragma once

#include <stdint.h>

namespace DeepestScatter 
{
    struct BatchSettings
    {
        explicit BatchSettings(int32_t batchStartId, int32_t batchSize): 
            batchStartId(batchStartId), batchSize(batchSize), batchEndId(batchStartId + batchSize)
        {
        }

        const int32_t batchStartId;
        const int32_t batchSize;
        const int32_t batchEndId;
    };
}
