#pragma once

#include <tuple>
#include <string>
#include <optixu/optixpp_namespace.h>
#include <iostream>

namespace DeepestScatter
{
    class Resources
    {
    public:
        Resources(std::shared_ptr<optix::Context> context) : context(*context.get()) {}

        /**
        * Load a volume from .vdb file with a density grid of type uint_8.
        * The values are scaled to be from 0 to 1.
        * Returns buffer and the bounding box of the cloud, in pixels.
        */
        std::tuple<optix::Buffer, optix::float3> loadVolumeBuffer(const std::string &path, bool createMipmaps = false);
        optix::Program loadProgram(const std::string& fileName, const std::string& programName);

    private:
        optix::Context context;
    };
}
