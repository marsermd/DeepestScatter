#pragma once

#include <vector>
#include <stdint.h>
#include <string>

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#pragma warning(push, 0)
#include <openvdb/openvdb.h>
#pragma warning(pop)

#include "Util/Arcball.h"

#include "VDBCloud.h"

namespace DeepestScatter
{
    class SceneItem;
    class Resources;

    class Scene final
    {
    public:
        struct Settings;

        Scene(const Settings& settings, const std::vector<std::shared_ptr<SceneItem>>& sceneItems, optix::Context context, 
            std::shared_ptr<Resources>& resources);

        void init();

        /**
        * Load cloud from .vdb file with a density grid of type float.
        */
        void addCloud(const std::string &path);

        void restartProgressive();

        void rotateCamera(optix::float2 from, optix::float2 to);
        void updateCamera();
        void display();

        void increaseExposure();
        void decreaseExposure();

        struct Settings
        {
            Settings(uint32_t width, uint32_t height, float sampleStep) :
                width(width), height(height), sampleStep(sampleStep) {}

            uint32_t width;
            uint32_t height;
            float sampleStep;
        };

    private:
        std::vector<std::shared_ptr<SceneItem>> sceneItems;
        std::shared_ptr<Resources> resources;

        float sampleStep;

        float cloudLengthMeters = 3000;
        float meanFreePathMeters = 10;

        uint32_t subframeId = 0;

        uint32_t width, height;
        optix::Context context;
        optix::Buffer  progressiveBuffer;
        optix::Buffer  varianceBuffer;
        optix::Buffer  screenBuffer;

        optix::float3         cameraUp;
        optix::float3         cameraLookat;
        optix::float3         cameraEye;
        optix::Matrix4x4      cameraRotate;

        sutil::Arcball arcball;
        optix::Program clearScreen;
        optix::Program exception;
        optix::Program miss;

        optix::Program camera;

        optix::Program reinhardFirstPass;
        optix::Program reinhardSecondPass;
        optix::Program reinhardLastPass;

        float_t exposure = 1.0f;
    };
}
