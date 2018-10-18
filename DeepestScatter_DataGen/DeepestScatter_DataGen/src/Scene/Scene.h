#pragma once

#include <vector>
#include <optixu/optixpp_namespace.h>

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
        Scene(const std::vector<std::shared_ptr<SceneItem>>& sceneItems, optix::Context context);

        void init();

        void restartProgressive();

        void update();

        bool isCompleted();
    private:
        std::vector<std::shared_ptr<SceneItem>> sceneItems;
        std::shared_ptr<Resources> resources;
        optix::Context context;
    };
}
