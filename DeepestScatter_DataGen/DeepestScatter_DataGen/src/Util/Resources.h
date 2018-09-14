#pragma once

#include <tuple>
#include <string>
#include <optixu/optixpp_namespace.h>

namespace DeepestScatter
{
    class Resources
    {
    public:
        Resources(optix::Context context) : context(context) {}

        /**
        * Load a volume from .vdb file with a density grid of type uint_8.
        * If there is a value that exceeds 1, we take 
        * The values are scaled to be from 0 to 1
        */
        optix::Buffer loadVolumeBuffer(const std::string &path);
        optix::Program loadProgram(const std::string &fileName, const std::string programName);

    private:
        optix::Context context;
    };
}