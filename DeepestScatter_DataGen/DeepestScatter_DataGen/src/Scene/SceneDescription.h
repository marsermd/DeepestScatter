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
                SunAndSkyAllScatter,
                SunMultipleScatter,
                SunSingleScatter
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
                Off = false,
                On = true
            };

            Model(const std::string& vdbPath, Mipmaps mipmaps, Size size)
                : vdbPath(vdbPath),
                  mipmapsOn(mipmaps),
                  size(size)
            {
            }

            const std::string vdbPath;
            const Mipmaps mipmapsOn;
            const Size size;
            const MeanFreePath meanFreePath = MeanFreePath{ Meter{10} };
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
}
