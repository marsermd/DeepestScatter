#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "Util/Arcball.h"
#include "SceneItem.h"

namespace DeepestScatter
{
    class Resources;

    class Camera : public SceneItem
    {
    public:
        struct Settings;

        Camera(const Settings& settings, optix::Context context, std::shared_ptr<Resources> resources):
            width(settings.width), height(settings.height),
            context(context),
            resources(resources) 
        {
        }

        virtual ~Camera() override = default;

        void init() override;
        void update() override;
        void reset() override;

        bool isCompleted() override;

        void rotate(optix::float2 from, optix::float2 to);

        void increaseExposure();
        void decreaseExposure();

        struct Settings
        {
            Settings(uint32_t width, uint32_t height) :
                width(width), height(height) {}

            uint32_t width;
            uint32_t height;
        };

        bool completed = true;

    private:
        void render();
        void updatePosition();

        optix::Context context;
        std::shared_ptr<Resources> resources;

        uint32_t width;
        uint32_t height;

        sutil::Arcball arcball;

        uint32_t subframeId = 0;

        optix::float3         cameraUp;
        optix::float3         cameraLookat;
        optix::float3         cameraEye;
        optix::Matrix4x4      cameraRotate;

        optix::Program clearScreen;
        optix::Program exception;
        optix::Program miss;

        optix::Program camera;

        float_t exposure = 1.0f;
        optix::Program reinhardFirstPass;
        optix::Program reinhardSecondPass;
        optix::Program reinhardLastPass;

        optix::Buffer  progressiveBuffer;
        optix::Buffer  varianceBuffer;
        optix::Buffer  screenBuffer;
    };
}