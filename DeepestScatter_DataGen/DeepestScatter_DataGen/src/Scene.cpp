#include "Scene.h"

#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <iostream>
#include <fstream>

#include <gsl/gsl_util>

#include <optix.h>
#include "GL/freeglut.h"

#pragma warning(push, 0)   
#include "../sutil/sutil.h"

#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>

#include <openvdb/Types.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/math/Stats.h>
#pragma warning(pop)

#include "Mie.h"


Scene::Scene(uint32_t width, uint32_t height, float sampleStep) :
    width(width), height(height), sampleStep(sampleStep)
{
    context = optix::Context::create();

    context["lightDirection"]->setFloat(-0.586f, -0.766f, -0.2717f);
    context["lightColor"]->setFloat(1.3f, 1.25f, 1.15f);
    context["lightIntensity"]->setFloat(0.6e9f);

    context["skyIntensity"]->setFloat(0, 0, 2000);
    context["groundIntensity"]->setFloat(600, 800, 1000);

    context->setRayTypeCount(1);
    context->setEntryPointCount(1);

    progressiveBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height);
    screenBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
}


Scene::~Scene()
{
}


void Scene::addCloud(const std::string &path)
{
    cloudBuffer = loadVolumetricData(path);
    cloudSampler = createSamplerForBuffer3D(cloudBuffer);
    context["cloud"]->setTextureSampler(cloudSampler);

    RTsize cloudX, cloudY, cloudZ;
    cloudBuffer->getSize(cloudX, cloudY, cloudZ);

    optix::Buffer inScatterOut = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE, cloudX, cloudY, cloudZ);

    optix::Program inScatter = loadProgram("inScatter.cu", "inScatter");
    inScatter["resultBuffer"]->setBuffer(inScatterOut);
    context->setRayGenerationProgram(0, inScatter);

    context->validate();
    context->launch(0, cloudX, cloudY, cloudZ);

    cloudSampler->destroy();
    cloudBuffer->destroy();

    inScatterIn = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, cloudX, cloudY, cloudZ);
    uint8_t* to = (uint8_t*)inScatterIn->map();
    uint8_t* from = (uint8_t*)inScatterOut->map();
    std::memcpy(to, from, sizeof(uint8_t) * cloudX * cloudY * cloudZ);
    inScatterIn->unmap();
    inScatterOut->unmap();

    inScatterOut->destroy();

    cloudBuffer = loadVolumetricData(path);
    cloudSampler = createSamplerForBuffer3D(cloudBuffer);
    context["cloud"]->setTextureSampler(cloudSampler);

    inScatterSampler = createSamplerForBuffer3D(inScatterIn);
    context["inScatter"]->setTextureSampler(inScatterSampler);

    cloudGeometry = context->createGeometry();
    cloudGeometry->setBoundingBoxProgram(loadProgram("cloud.cu", "bounds"));
    cloudGeometry->setIntersectionProgram(loadProgram("cloud.cu", "intersect"));
    cloudGeometry->setPrimitiveCount(1u);
    cloudGeometry["minimalRayDistance"]->setFloat(0.001f);

    cloudMaterial = context->createMaterial();
    cloudMaterial->setClosestHitProgram(0, loadProgram("cloud.cu", "closestHitRadiance"));

    cloudInstance = context->createGeometryInstance(cloudGeometry, &cloudMaterial, &cloudMaterial + 1);
    cloudGroup = context->createGeometryGroup();
    cloudGroup->addChild(cloudInstance);
    cloudGroup->setAcceleration(context->createAcceleration("MedianBvh", "Bvh"));

    context["objectRoot"]->set(cloudGroup);

    context->setRayGenerationProgram(0, camera);
    context["radianceRayType"]->setUint(0u);
}

void Scene::restartProgressive()
{
    std::cout << "restarting progressive" << std::endl;
    subframeId = 0;
    context->setRayGenerationProgram(0, clearScreen);
    context->launch(0, width, height);
}

void Scene::init()
{
    context["sampleStep"]->setFloat(sampleStep);
    context["densityMultiplier"]->setFloat(cloudLengthMeters / meanFreePathMeters);

    context["mie"]->setTextureSampler(Mie::getMieSampler(context));
    context["choppedMie"]->setTextureSampler(Mie::getChoppedMieSampler(context));
    context["choppedMieIntegral"]->setTextureSampler(Mie::getChoppedMieIntegralSampler(context));

    context["progressiveBuffer"]->setBuffer(progressiveBuffer);

    camera = loadProgram("pinholeCamera.cu", "pinholeCamera");

    clearScreen = loadProgram("pinholeCamera.cu", "pinholeCamera");

    cameraEye = optix::make_float3(2, -0.4f, 0);
    cameraLookat = optix::make_float3(0, 0, 0);
    cameraUp = optix::make_float3(0, 1, 0);

    cameraRotate = optix::Matrix4x4::identity();
    updateCamera();

    exception = loadProgram("pinholeCamera.cu", "exception");
    context->setExceptionProgram(0, exception);
    context["errorColor"]->setFloat(1, 0.6f, 0.6f);

    miss = loadProgram("pinholeCamera.cu", "miss");
    context->setMissProgram(0, miss);
    context["missColor"]->setFloat(900, 1000, 1600);

    reinhardFirstPass = loadProgram("reinhard.cu", "firstPass");
    reinhardSecondPass = loadProgram("reinhard.cu", "secondPass");
    reinhardLastPass = loadProgram("reinhard.cu", "applyReinhard");

    context["sumLogColumns"]->setBuffer(context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, width));
    context["lAverage"]->setBuffer(context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, 1));
    context["totalPixels"]->setUint(width * height);
    context["screenBuffer"]->setBuffer(screenBuffer);
}

void Scene::rotateCamera(optix::float2 from, optix::float2 to)
{
    cameraRotate = arcball.rotate(from, to);
    updateCamera();
    restartProgressive();
}

void Scene::updateCamera()
{
    const float vfov = 30.0f;
    const float aspectRatio = static_cast<float>(width) /
        static_cast<float>(height);

    optix::float3 u, v, w;
    sutil::calculateCameraVariables(
        cameraEye, cameraLookat, cameraUp, vfov, aspectRatio,
        u, v, w);

    const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
        normalize(u),
        normalize(v),
        normalize(w),
        cameraLookat);
    const optix::Matrix4x4 transform = frame * cameraRotate * frame.inverse();

    cameraEye = make_float3(transform * make_float4(cameraEye, 1.0f));

    sutil::calculateCameraVariables(
        cameraEye, cameraLookat, cameraUp, vfov, aspectRatio,
        u, v, w, true);

    cameraRotate = optix::Matrix4x4::identity();

    camera["eye"]->setFloat(cameraEye);
    camera["U"]->setFloat(u);
    camera["V"]->setFloat(v);
    camera["W"]->setFloat(w);
}


void Scene::display()
{
    context->validate();

    for (int i = 0; i < 10; i++)
    {
        subframeId++;
        context["subframeId"]->setUint(subframeId);
        context->setRayGenerationProgram(0, camera);
        context->launch(0, width, height);

        context->setRayGenerationProgram(0, reinhardFirstPass);
        context->launch(0, width, 1);

        context->setRayGenerationProgram(0, reinhardSecondPass);
        context->launch(0, 1, 1);

        reinhardLastPass["exposure"]->setFloat(exposure);
        context->setRayGenerationProgram(0, reinhardLastPass);
        context->launch(0, width, height);
    }

    GLenum glDataType = GL_UNSIGNED_BYTE;
    GLenum glFormat = GL_RGBA;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    
    GLvoid* imageData = screenBuffer->map();
    glDrawPixels(width, height, glFormat, glDataType, imageData);
    screenBuffer->unmap();
}

void Scene::increaseExposure()
{
    exposure *= 1.2f;
}

void Scene::decreaseExposure()
{
    exposure /= 1.2f;
}

optix::Buffer Scene::loadVolumetricData(const std::string &path)
{
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

    float maxSize = gsl::narrow<float>(std::max({ sizeX, sizeY, sizeZ }));
    context["boxSize"]->setFloat(sizeX / maxSize, sizeY / maxSize, sizeZ / maxSize);

    std::cout << "Cloud's size is: " << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;

    std::cout << "Creating buffer" << std::endl;

    optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE);
    buffer->setSize(sizeX, sizeY, sizeZ);

    {
        uint8_t* density = (uint8_t*)buffer->map();

        std::memset(density, 0, sizeof(uint8_t) * sizeX * sizeY * sizeZ);

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

        buffer->unmap();
    }

    return buffer;
}

optix::TextureSampler Scene::createSamplerForBuffer3D(optix::Buffer buffer)
{
    optix::TextureSampler sampler3D = context->createTextureSampler();

    for (uint32_t dim = 0; dim < 3; dim++)
    {
        sampler3D->setWrapMode(dim, RT_WRAP_CLAMP_TO_EDGE);
    }

    sampler3D->setFilteringModes(
        RT_FILTER_LINEAR,
        RT_FILTER_LINEAR,
        RT_FILTER_NONE
    );

    sampler3D->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler3D->setBuffer(buffer);

    return sampler3D;
}

optix::Program Scene::loadProgram(const std::string &fileName, const std::string programName)
{
    const std::string ptxPath = "./CUDA/";
    const std::string ptxExtension = ".ptx";
    
    std::string path = ptxPath + fileName + ptxExtension;

    std::cout << "loading program " << path << std::endl;

    return context->createProgramFromPTXFile(path, programName);
}