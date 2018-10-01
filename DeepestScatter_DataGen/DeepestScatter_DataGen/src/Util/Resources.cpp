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
#pragma warning(pop)

namespace DeepestScatter
{
    size_t getId(size_t x, size_t y, size_t z, size_t size)
    {
        return z * size * size + y * size + x;
    }

    std::tuple<optix::Buffer, optix::float3> Resources::loadVolumeBuffer(const std::string &path, bool createMipmaps)
    {
        std::cout << "Loading VDB" << path << std::endl;

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

        double maxDensity = ex.max();

        openvdb::CoordBBox boundingBox = grid->evalActiveVoxelBoundingBox();
        boundingBox.expandBy(1);
        openvdb::Coord min = boundingBox.min();
        openvdb::Coord max = boundingBox.max() + openvdb::Coord(1, 1, 1);
        openvdb::Coord boundingSize = max - min;

        size_t sizeX = boundingSize.x();
        size_t sizeY = boundingSize.y();
        size_t sizeZ = boundingSize.z();

        std::cout << "Creating buffer of size " << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;

        auto buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE);
        int levelCount = 1;
        if (createMipmaps)
        {
            size_t maxSize = std::max({ sizeX, sizeY, sizeZ });
            levelCount = 1;
            while (maxSize >>= 1)
            {
                levelCount++;
            }
            sizeX = sizeY = sizeZ = 1 << levelCount;
        }
        buffer->setMipLevelCount(levelCount);
        buffer->setSize(sizeX, sizeY, sizeZ);

        {
            uint8_t* density = (uint8_t*)buffer->map(0);

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

            buffer->unmap(0);
        }

        if (createMipmaps)
        {
            size_t maxSize = std::max({ sizeX, sizeY, sizeZ });

            size_t size = sizeX;
            for (int level = 1; level < levelCount; level++)
            {
                size >>= 1;

                uint8_t* prevLevel = (uint8_t*)buffer->map(level - 1);
                uint8_t* curLevel = (uint8_t*)buffer->map(level);

                for (uint32_t z = 0; z < size; z++)
                {
                    for (uint32_t y = 0; y < size; y++)
                    {
                        for (uint32_t x = 0; x < size; x++)
                        {
                            uint16_t curValue =
                                (uint16_t)prevLevel[getId(x * 2    , y * 2    , z * 2    , size * 2)] +
                                (uint16_t)prevLevel[getId(x * 2    , y * 2    , z * 2 + 1, size * 2)] +
                                (uint16_t)prevLevel[getId(x * 2    , y * 2 + 1, z * 2    , size * 2)] +
                                (uint16_t)prevLevel[getId(x * 2    , y * 2 + 1, z * 2 + 1, size * 2)] +
                                (uint16_t)prevLevel[getId(x * 2 + 1, y * 2    , z * 2    , size * 2)] +
                                (uint16_t)prevLevel[getId(x * 2 + 1, y * 2    , z * 2 + 1, size * 2)] +
                                (uint16_t)prevLevel[getId(x * 2 + 1, y * 2 + 1, z * 2    , size * 2)] +
                                (uint16_t)prevLevel[getId(x * 2 + 1, y * 2 + 1, z * 2 + 1, size * 2)];
                            curValue /= 8;
                            curLevel[getId(x, y, z, size)] = gsl::narrow<uint8_t>(curValue);
                        }
                    }
                }

                buffer->unmap(level);
                buffer->unmap(level - 1);
            }
            assert(size == 1);
        }

        optix::float3 floatSize = optix::make_float3(boundingSize.x(), boundingSize.y(), boundingSize.z());
        return std::make_tuple(buffer, floatSize);
    }

    optix::Program Resources::loadProgram(const std::string &fileName, const std::string programName)
    {
        const static std::string ptxPath = "./CUDA/";
        const static std::string ptxExtension = ".ptx";

        std::string path = ptxPath + fileName + ptxExtension;

        std::cout << "Loading program: " << path << "::" << programName << std::endl;


        return context->createProgramFromPTXFile(path, programName);
    }
}
