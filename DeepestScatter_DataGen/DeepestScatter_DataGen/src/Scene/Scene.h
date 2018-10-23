#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <optixu/optixpp_namespace.h>

#include "SceneItem.h"

namespace DeepestScatter
{
    class SceneItem;
    class Resources;

    class Scene final
    {
    public:
        Scene(std::vector<std::shared_ptr<SceneItem>> sceneItems, std::shared_ptr<optix::Context> context);

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
