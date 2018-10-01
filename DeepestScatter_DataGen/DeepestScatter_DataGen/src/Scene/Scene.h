#pragma once

#include <vector>
#include <stdint.h>
#include <string>

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#pragma warning(push, 0)
#include <openvdb/openvdb.h>
#pragma warning(pop)

#include "VDBCloud.h"

namespace DeepestScatter
{
    class SceneItem;
    class Resources;

    class Scene final
    {
    public:
        using SampleStep = float;

        Scene(SampleStep sampleStep, const std::vector<std::shared_ptr<SceneItem>>& sceneItems, optix::Context context);

        void Init();

        void RestartProgressive();

        void Display();
    private:
        std::vector<std::shared_ptr<SceneItem>> sceneItems;
        std::shared_ptr<Resources> resources;

        float sampleStep;

        float cloudLengthMeters = 3000;
        float meanFreePathMeters = 10;

        optix::Context context;
    };
}
