#pragma once
#include <optixu/optixpp_namespace.h>

namespace DeepestScatter
{
    class ARenderer
    {
    public:
        ARenderer() = default;

        virtual ~ARenderer() = default;

        virtual optix::Program getCamera() = 0;
        virtual void init() = 0;
        virtual void render(optix::Buffer frameResultBuffer) = 0;
    };
}
