#pragma once

#include <optixu/optixpp_namespace.h>

namespace DeepestScatter
{
    namespace Mie
    {
        optix::TextureSampler getMieSampler(optix::Context context);
        optix::TextureSampler getChoppedMieSampler(optix::Context context);
        optix::TextureSampler getChoppedMieIntegralSampler(optix::Context context);
    }
}
