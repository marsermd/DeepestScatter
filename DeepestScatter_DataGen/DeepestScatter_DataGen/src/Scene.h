#pragma once

#include <stdint.h>
#include <string>
#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "../sutil/Arcball.h"

class Scene
{
public:
    Scene(uint32_t width, uint32_t height, float sampleStep);
    ~Scene();

    void init();
    /**
    * Load cloud from binary file that consists of
    * sizeX:uint32, sizeY:uint32, sizeZ:uint32, density:float32[sizeX * sizeY * sizeZ]
    */
    void addCloud(const std::string &path);

    void restartProgressive();

    void rotateCamera(optix::float2 from, optix::float2 to);
    void updateCamera();
    void display();

    void increaseExposure();
    void decreaseExposure();

private:
    float sampleStep;

    float cloudLengthMeters = 3000;
    float meanFreePathMeters = 10;

    uint32_t subframeId = 0;

    uint32_t width, height;
    optix::Context context;
    optix::Buffer  progressiveBuffer;
    optix::Buffer  screenBuffer;

    optix::Buffer           cloudBuffer;
    optix::TextureSampler   cloudSampler;
    optix::Geometry         cloudGeometry;
    optix::Material         cloudMaterial;
    optix::GeometryInstance cloudInstance;
    optix::GeometryGroup    cloudGroup;
    optix::Buffer           inScatterIn;
    optix::TextureSampler   inScatterSampler;

    optix::float3         cameraUp;
    optix::float3         cameraLookat;
    optix::float3         cameraEye;
    optix::Matrix4x4      cameraRotate;

    sutil::Arcball arcball;
    optix::Program camera;
    optix::Program clearScreen;
    optix::Program exception;
    optix::Program miss;

    optix::Program reinhardFirstPass;
    optix::Program reinhardSecondPass;
    optix::Program reinhardLastPass;

    float_t midGrey = 0.0003f;

    /**
    * Read from binary file that consists of
    * sizeX:uint32, sizeY:uint32, sizeZ:uint32, density:float32[sizeX * sizeY * sizeZ]
    */
    optix::Buffer loadVolumetricData(const std::string &path);

    optix::TextureSampler createSamplerForBuffer3D(optix::Buffer buffer);

    optix::Program loadProgram(
        const std::string &fileName,
        const std::string programName);
};

