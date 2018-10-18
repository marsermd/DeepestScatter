#pragma once
#include "SceneItem.h"
#include <internal/optix_datatypes.h>
#include "SceneDescription.h"

namespace DeepestScatter
{
    class Sun: public SceneItem
    {
    public:
        typedef DirectionalLight Settings;

        Sun(Settings settings, optix::Context context);

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
