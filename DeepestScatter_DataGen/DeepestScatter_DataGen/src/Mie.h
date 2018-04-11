#pragma once

#include <optix.h>
#include <optixu/optixpp_namespace.h>

namespace Mie
{
    optix::TextureSampler getMieSampler(optix::Context context);
    optix::TextureSampler getChoppedMieSampler(optix::Context context);
    optix::TextureSampler getChoppedMieIntegralSampler(optix::Context context);
}
