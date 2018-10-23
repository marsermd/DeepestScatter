#include "Installers.h"

#include "Scene/SceneDescription.h"

#include <memory>

#include "Scene/Sun.h"
#include "Scene/VDBCloud.h"
#include "Scene/CloudPTRenderer.h"
#include "Scene/SceneSetupCollector.h"
#include "Scene/Camera.h"

#include "Scene/Scene.h"

namespace DeepestScatter
{
    template<class T>
    void addSceneItem(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<T>().as<SceneItem>().asSelf().singleInstance();
    }

    Hypodermic::ContainerBuilder installPathTracingApp()
    {
        namespace di = Hypodermic;
        di::ContainerBuilder builder;

        addSceneItem<Sun>(builder);
        addSceneItem<VDBCloud>(builder);
        addSceneItem<CloudPTRenderer>(builder);
        addSceneItem<Camera>(builder);

        return builder;
    }

    Hypodermic::ContainerBuilder installSetupCollectorApp()
    {
        namespace di = Hypodermic;
        di::ContainerBuilder builder;

        addSceneItem<Sun>(builder);
        addSceneItem<VDBCloud>(builder);
        addSceneItem<CloudPTRenderer>(builder);
        addSceneItem<SceneSetupCollector>(builder);
        addSceneItem<Camera>(builder);

        return builder;
    }

    Hypodermic::ContainerBuilder installFramework(
        const std::string& cloudPath,
        const std::string& databasePath,
        uint32_t width, uint32_t height)
    {
        namespace di = Hypodermic;

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

        di::ContainerBuilder builder;

        builder.addRegistrations(BindSceneDescription(scene));
        builder.registerInstance(std::make_shared<Dataset::Settings>(databasePath));
        builder.registerInstance(std::make_shared<Camera::Settings>(width, height));
        builder.registerInstance(std::make_shared<BatchSettings>(2048));

        builder.registerInstance(std::make_shared<optix::Context>(optix::Context::create()));

        builder.registerType<Resources>().singleInstance();
        builder.registerType<Dataset>().singleInstance();
        builder.registerType<Scene>().singleInstance();

        return builder;
    }
}
