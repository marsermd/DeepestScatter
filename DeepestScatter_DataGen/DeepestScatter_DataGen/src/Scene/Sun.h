#pragma once
#include "SceneItem.h"
#include "SceneDescription.h"
#include <optix_world.h>

namespace DeepestScatter
{
    class Sun: public SceneItem
    {
    public:
        typedef DirectionalLight Settings;

        Sun(std::shared_ptr<Settings> settings, std::shared_ptr<optix::Context> context);

        void init() override;
        void reset() override {}
        void update() override {}

    private:
        optix::Context context;

        optix::float3 direction;
        Color color;
        float intensity;
    };
}
