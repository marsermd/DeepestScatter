#pragma once

#include <memory>
#include <utility>

#include "Util/Resources.h"
#include "Util/Arcball.h"

#include "Scene/SceneItem.h"
#include "ARenderer.h"

namespace DeepestScatter
{
    class Resources;

    class Camera : public SceneItem
    {
    public:
        struct Settings
        {
            Settings(uint32_t width, uint32_t height) :
                width(width), height(height) {}

            uint32_t width;
            uint32_t height;
        };

        Camera(
            std::shared_ptr<Settings> settings, 
            std::shared_ptr<optix::Context> context,
            std::shared_ptr<Resources> resources,
            std::shared_ptr<ARenderer> renderer) :
            width(settings->width), height(settings->height),
            context(*context.get()),
            resources(std::move(resources)),
            renderer(std::move(renderer))
        {
        }

        virtual ~Camera() noexcept = default;

        void init() override;
        void update() override;
        void reset() override;

        bool isCompleted() override;

        void rotate(optix::float2 from, optix::float2 to);

        void increaseExposure();
        void decreaseExposure();

        bool completed = true;

    private:
        void setupVariables(optix::Program& program);

        void saveToDisk();

        void render();

        void updatePosition();

        bool isConverged();

        uint32_t width;
        uint32_t height;

        optix::Context context;
        std::shared_ptr<Resources> resources;
        std::shared_ptr<ARenderer> renderer;

        sutil::Arcball arcball;

        uint32_t subframeId = 0;

        optix::float3         cameraUp{};
        optix::float3         cameraLookat{};
        optix::float3         cameraEye{};
        optix::Matrix4x4      cameraRotate{};

        optix::Program clearScreen;
        optix::Program exception;
        optix::Program miss;
        optix::Program updateFrameResult;

        float_t exposure = 0.4f;
        optix::Program reinhardFirstPass;
        optix::Program reinhardSecondPass;
        optix::Program reinhardLastPass;

        optix::Buffer  reinhardSumLuminanceColumn;
        optix::Buffer  reinhardAverageLuminance;

        optix::Buffer  frameResultBuffer;
        optix::Buffer  progressiveBuffer;
        optix::Buffer  varianceBuffer;
        optix::Buffer  screenBuffer;
    };
}
