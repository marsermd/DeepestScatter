#pragma once

#include "ARenderer.h"

namespace DeepestScatter
{
    class EmptyRenderer: public ARenderer
    {
    public:
        EmptyRenderer() = default;
        virtual ~EmptyRenderer() = default;

        virtual optix::Program getCamera() override;
        virtual void init() override {};
        virtual void render(optix::Buffer frameResultBuffer) override {};
    };
}
