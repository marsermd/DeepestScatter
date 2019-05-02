#pragma once

#include <tuple>
#include <string>
#include <optixu/optixpp_namespace.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <optix_sizet.h>

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
        class VolumeCache
        {
        public:
            const std::string cloudPath;
            const optix::float3 floatSize;

            VolumeCache(std::string path, optix::Buffer& buffer, optix::float3 floatSize);

            void fillBuffer(optix::Buffer& buffer);
        private:
            
            optix::size_t3 size;
            std::vector<std::vector<uint8_t>> cache{};
        };

        optix::Context context;

        static std::unique_ptr<VolumeCache> volumeCache;
    };
}
