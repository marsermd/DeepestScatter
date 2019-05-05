#include "Resources.h"

#include <iostream>
#include <fstream>
#include <gsl/gsl_util>
#include <gsl/span>

#include <optixu/optixu_math_namespace.h>

#pragma warning(push, 0)
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>

#include <openvdb/Types.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/math/Stats.h>
#include "BufferBind.h"
#pragma warning(pop)

namespace DeepestScatter
{
    std::unique_ptr<Resources::VolumeCache> Resources::volumeCache = nullptr;

    class TextureView3D
    {
    public:
        TextureView3D(uint8_t* values, size_t sizeX, size_t sizeY, size_t sizeZ)
            : values(values), sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ)
        {
        }

        uint16_t get(size_t x, size_t y, size_t z) const
        {
            if (isOutOfRange(x, sizeX) ||
                isOutOfRange(y, sizeY) ||
                isOutOfRange(z, sizeZ))
            {
                return 0;
            }
            const size_t id = z * sizeX * sizeY + y * sizeX + x;
            return values[id];
        }

        void set(size_t x, size_t y, size_t z, uint8_t value) const
        {
            if (isOutOfRange(x, sizeX) ||
                isOutOfRange(y, sizeY) ||
                isOutOfRange(z, sizeZ))
            {
                throw std::out_of_range("Out of texture range");
            }
            const size_t id = z * sizeX * sizeY + y * sizeX + x;
            values[id] = value;
        }

    private:
        uint8_t* values;
        const size_t sizeX;
        const size_t sizeY;
        const size_t sizeZ;

        static bool isOutOfRange(int32_t value, size_t size)
        {
            return value < 0 || value >= size;
        }
    };

    std::tuple<optix::Buffer, optix::float3> Resources::loadVolumeBuffer(const std::string &path, bool createMipmaps)
    {
        std::cout << "Loading VDB... " << path << std::endl;
        auto buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE);

        if (volumeCache != nullptr && volumeCache->cloudPath == path)
        {
            std::cout << "Using cached." << std::endl;
            volumeCache->fillBuffer(buffer);
            return std::make_tuple(buffer, volumeCache->floatSize);;
        }

        openvdb::initialize();

        using GridType = openvdb::FloatGrid;
        std::ifstream ifile(path, std::ios_base::binary);

        assert(ifile.good());

        auto grids = openvdb::io::Stream(ifile).getGrids();
        auto grid = openvdb::gridPtrCast<GridType>((*grids)[0]);
        auto gridAccessor = grid->getConstUnsafeAccessor();

        using LeafIterType = GridType::ValueOnCIter;
        const auto addOp = [](const LeafIterType& iter, openvdb::math::Extrema& ex) { ex.add(*iter); };
        openvdb::math::Extrema ex = openvdb::tools::extrema(grid->tree().cbeginValueOn(), addOp, /*threaded=*/true);

        const double maxDensity = ex.max();

        openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();
        boundingBox = boundingBox.expandBy(1);
        openvdb::Coord min = boundingBox.min();
        openvdb::Coord max = boundingBox.max() + openvdb::Coord(1, 1, 1);
        openvdb::Coord boundingSize = max - min;

        size_t sizeX = boundingSize.x();
        size_t sizeY = boundingSize.y();
        size_t sizeZ = boundingSize.z();

        int levelCount = 1;
        if (createMipmaps)
        {
            size_t maxSize = std::max({ sizeX, sizeY, sizeZ });
            levelCount = 1;
            while (maxSize /= 2)
            {
                levelCount++;
            }
            std::cout << "And creating mipmaps... " << levelCount << std::endl;
        }

        std::cout << "Creating buffer of size " << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;
        buffer->setMipLevelCount(levelCount);
        buffer->setSize(sizeX, sizeY, sizeZ);

        {
            BufferBind<uint8_t> density(buffer, 0);

            uint32_t targetPos = 0;
            for (int32_t z = 0; z < sizeZ; z++)
            {
                for (int32_t y = 0; y < sizeY; y++)
                {
                    for (int32_t x = 0; x < sizeX; x++)
                    {
                        auto pos = openvdb::Coord(
                            min.x() + x, 
                            min.y() + y, 
                            min.z() + z);
                        density[targetPos] = gsl::narrow_cast<uint8_t>(gridAccessor.getValue(pos) / maxDensity * 255);
                        targetPos++;
                    }
                }
            }
            assert(targetPos == sizeX * sizeY * sizeZ);
        }

        if (createMipmaps)
        {
            generateMipmaps(buffer);
        }

        optix::float3 floatSize = optix::make_float3(boundingSize.x(), boundingSize.y(), boundingSize.z());

        volumeCache = std::make_unique<VolumeCache>(path, buffer, floatSize);

        return std::make_tuple(buffer, floatSize);
    }

    optix::Program Resources::loadProgram(const std::string& fileName, const std::string& programName)
    {
        const static std::string ptxPath = "./CUDA/";
        const static std::string ptxExtension = ".ptx";

        const std::string path = ptxPath + fileName + ptxExtension;

        std::cout << "Loading program: " << path << "::" << programName << std::endl;

        return context->createProgramFromPTXFile(path, programName);
    }

    void Resources::generateMipmaps(optix::Buffer buffer) const
    {
        const uint32_t levelCount = buffer->getMipLevelCount();

        size_t sizeX, sizeY, sizeZ;
        buffer->getSize(sizeX, sizeY, sizeZ);

        for (uint32_t level = 1; level < levelCount; level++)
        {
            assert(sizeX != 1 || sizeY != 1 || sizeZ != 1);
            BufferBind<uint8_t> prevLevel(buffer, level - 1);
            BufferBind<uint8_t> curLevel(buffer, level);

            TextureView3D prevView(&prevLevel[0], sizeX, sizeY, sizeZ);

            buffer->getMipLevelSize(level, sizeX, sizeY, sizeZ);
            TextureView3D curView(&curLevel[0], sizeX, sizeY, sizeZ);

            for (uint32_t z = 0; z < sizeZ; z++)
            {
                for (uint32_t y = 0; y < sizeY; y++)
                {
                    for (uint32_t x = 0; x < sizeX; x++)
                    {
                        uint16_t curValue =
                            prevView.get(x * 2, y * 2, z * 2) +
                            prevView.get(x * 2, y * 2, z * 2 + 1) +
                            prevView.get(x * 2, y * 2 + 1, z * 2) +
                            prevView.get(x * 2, y * 2 + 1, z * 2 + 1) +
                            prevView.get(x * 2 + 1, y * 2, z * 2) +
                            prevView.get(x * 2 + 1, y * 2, z * 2 + 1) +
                            prevView.get(x * 2 + 1, y * 2 + 1, z * 2) +
                            prevView.get(x * 2 + 1, y * 2 + 1, z * 2 + 1);
                        curValue /= 8;
                        curView.set(x, y, z, gsl::narrow<uint8_t>(curValue));
                    }
                }
            }
        }
        assert(sizeX == 1 && sizeY == 1 && sizeZ == 1);
    }

    Resources::VolumeCache::VolumeCache(std::string cloudPath, optix::Buffer& buffer, optix::float3 floatSize):
        cloudPath(cloudPath),
        floatSize(floatSize)
    {
        std::cout << "Creating cache" << std::endl;
        const size_t mipLevelCount = buffer->getMipLevelCount();

        buffer->getSize(size.x, size.y, size.z);

        for (size_t i = 0; i < mipLevelCount; i++)
        {
            optix::size_t3 levelSize;
            buffer->getMipLevelSize(i, levelSize.x, levelSize.y, levelSize.z);
            BufferBind<uint8_t> bufferLevel(buffer, i);

            const size_t totalSize = levelSize.x * levelSize.y * levelSize.z;

            std::vector<uint8_t> level(totalSize);
            std::memcpy(&level[0], &bufferLevel[0], totalSize * sizeof(uint8_t));

            cache.emplace_back(level);
        }
    }

    Resources::VolumeCache::~VolumeCache()
    {
    }

    void Resources::VolumeCache::fillBuffer(optix::Buffer& buffer)
    {
        const size_t mipLevelCount = cache.size();
        buffer->setSize(size.x, size.y, size.z);
        buffer->setMipLevelCount(mipLevelCount);
        for (int i = 0; i < mipLevelCount; i++)
        {
            BufferBind<uint8_t> bufferLevel(buffer, i);
            const std::vector<uint8_t>& level = cache[i];

            const size_t totalSize = level.size();

            std::memcpy(&bufferLevel[0], &cache[i][0], totalSize * sizeof(uint8_t));
        }
    }
}
