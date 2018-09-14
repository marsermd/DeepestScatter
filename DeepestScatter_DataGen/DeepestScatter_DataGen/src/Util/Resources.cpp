#include "Resources.h"

#include <iostream>
#include <fstream>
#include <gsl/gsl_util>

#pragma warning(push, 0)
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>

#include <openvdb/Types.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/math/Stats.h>
#pragma warning(pop)

namespace DeepestScatter
{
    optix::Buffer Resources::loadVolumeBuffer(const std::string &path)
    {
        std::cout << "Loading VDB" << path << std::endl;

        openvdb::initialize();

        using GridType = openvdb::FloatGrid;
        std::ifstream ifile(path, std::ios_base::binary);

        assert(ifile.good());

        auto grids = openvdb::io::Stream(ifile).getGrids();
        auto grid = openvdb::gridPtrCast<GridType>((*grids)[0]);
        auto gridAccessor = grid->getConstAccessor();

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
        buffer->setSize(sizeX, sizeY, sizeZ);

        {
            uint8_t* density = (uint8_t*)buffer->map();

            uint32_t targetPos = 0;
            for (int32_t z = min.z(); z < max.z(); z++)
            {
                for (int32_t y = min.y(); y < max.y(); y++)
                {
                    for (int32_t x = min.x(); x < max.x(); x++)
                    {
                        density[targetPos] = (uint8_t)(gridAccessor.getValue(openvdb::Coord(x, y, z)) / maxDensity * 255);
                        targetPos++;
                    }
                }
            }
            assert(targetPos == sizeX * sizeY * sizeZ);

            buffer->unmap();
        }

        return buffer;
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
