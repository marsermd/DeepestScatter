#pragma once

#include "Scene/SceneDescription.h"

#include "Scene/CloudPTRenderer.h"
#include "Scene/VDBCloud.h"
#include "Scene/Scene.h"
#include "Scene/Sun.h"
#include "Boost/DIConfig.h"
#include "Scene/SceneSetupCollector.h"

namespace DeepestScatter
{
    inline auto installPathTracingApp()
    {
        namespace di = boost::di;
        return di::make_injector<DIConfig>
        (
            di::bind<SceneItem*[]>.to
            <
                Sun,
                VDBCloud,
                CloudPTRenderer,
                Camera
            >()
        );
    }

    inline auto installSetupCollectorApp()
    {
        namespace di = boost::di;
        return di::make_injector<DIConfig>
        (
            di::bind<SceneItem*[]>.to
            <
            Sun,
            VDBCloud,
            CloudPTRenderer,
            SceneSetupCollector,
            Camera
            >()
        );
    }

    inline auto installFramework(
        std::string& cloudPath,
        std::string& databasePath,
        uint32_t width, uint32_t height)
    {
        namespace di = boost::di;

        auto scene = SceneDescription
        {
            Cloud
            {
                Cloud::Rendering
                {
                    Cloud::Rendering::SampleStep{1.0f / 512.f},
                    Cloud::Rendering::Mode::Full
                },
                Cloud::Model
                {
                    cloudPath,
                    Cloud::Model::Mipmaps::Off,
                    Cloud::Model::Size{Meter{3000}},
                    Cloud::Model::MeanFreePath{Meter{10}}
                }
            },
            DirectionalLight
            {
                optix::make_float3(-0.586f, -0.766f, -0.2717f),
                Color{optix::make_float3(1.3f, 1.25f, 1.15f)},
                6e5f
            }
        };

        auto injector = di::make_injector<DIConfig>
        (
            di::bind<optix::Context>.to(optix::Context::create()),

            BindSceneDescription(scene),
            di::bind<Dataset::Settings>.to(Dataset::Settings(databasePath)),
            di::bind<Camera::Settings>.to(Camera::Settings(width, height)),
            di::bind<BatchSettings>.to(BatchSettings(2048)),

            di::bind<Resources>.to<Resources>(),

            di::bind<Scene>.to<Scene>()
        );
        return injector;
    }
}
