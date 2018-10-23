#pragma once

#include "Hypodermic/Hypodermic.h"

#include <memory>
#include <optixu/optixu_math_namespace.h>

namespace DeepestScatter
{
    using Meter = float;
    using Color = optix::float3;

    struct DirectionalLight
    {
        DirectionalLight(const optix::float3& direction, const Color& color, float intensity)
            : direction(optix::normalize(direction)),
              color(color),
              intensity(intensity)
        {
        }

        const optix::float3 direction;
        const Color color;
        const float intensity;
    };

    struct Cloud
    {
        struct Rendering;
        struct Model;

        Cloud(const Rendering& rendering, const Model& model)
            : rendering(rendering),
              model(model)
        {
        }

        struct Rendering
        {
            using SampleStep = float;

            enum class Mode
            {
                Full,
                SunMultipleScatter
            };

            Rendering(SampleStep sampleStep, Mode mode)
                : sampleStep(sampleStep),
                  mode(mode)
            {
            }

            const SampleStep sampleStep;
            const Mode mode;
        };

        struct Model
        {
            using MeanFreePath = Meter;
            using Size = Meter;

            enum class Mipmaps : bool
            {
                Off,
                On
            };

            Model(const std::string& vdbPath, Mipmaps mipmaps, Size size, Meter meanFreePath)
                : vdbPath(vdbPath),
                  mipmapsOn(mipmaps),
                  size(size),
                  meanFreePath(meanFreePath)
            {
            }

            const std::string vdbPath;
            const Mipmaps mipmapsOn;
            const Size size;
            const MeanFreePath meanFreePath;
        };

        const Rendering rendering;
        const Model model;
    };

    struct SceneDescription
    {
        SceneDescription(const Cloud& cloud, const DirectionalLight& light)
            : cloud(cloud),
              light(light)
        {
        }

        const Cloud cloud;
        const DirectionalLight light;
    };

    inline auto BindSceneDescription(SceneDescription& scene)
    {
        namespace di = Hypodermic;
        di::ContainerBuilder builder;

        builder.registerInstance(std::make_shared<Cloud::Model>(scene.cloud.model));
        builder.registerInstance(std::make_shared<Cloud::Rendering>(scene.cloud.rendering));
        builder.registerInstance(std::make_shared<Cloud>(scene.cloud));

        builder.registerInstance(std::make_shared<DirectionalLight>(scene.light));

        builder.registerInstance(std::make_shared<SceneDescription>(scene));

        return builder;
    }
}
