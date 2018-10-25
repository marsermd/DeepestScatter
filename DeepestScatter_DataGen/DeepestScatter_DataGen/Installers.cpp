#include "Installers.h"

#include "Scene/SceneDescription.h"

#include <memory>

#include "Scene/Sun.h"
#include "Scene/VDBCloud.h"
#include "Scene/CloudPTRenderer.h"
#include "Scene/SceneSetupCollector.h"
#include "Scene/Camera.h"

#include "Scene/Scene.h"
#include "Scene/RadianceCollector.h"

namespace DeepestScatter
{
    namespace di = Hypodermic;

    template<class T>
    void addSceneItem(di::ContainerBuilder& builder)
    {
        builder.registerType<T>().template as<SceneItem>().asSelf().singleInstance();
    }

    di::ContainerBuilder installPathTracingApp()
    {
        di::ContainerBuilder builder;

        addSceneItem<Sun>(builder);
        addSceneItem<VDBCloud>(builder);
        addSceneItem<CloudPTRenderer>(builder);
        addSceneItem<Camera>(builder);

        return builder;
    }

    di::ContainerBuilder installSetupCollectorApp()
    {
        di::ContainerBuilder builder;

        addSceneItem<Sun>(builder);
        addSceneItem<VDBCloud>(builder);
        addSceneItem<CloudPTRenderer>(builder);
        addSceneItem<SceneSetupCollector>(builder);
        addSceneItem<Camera>(builder);

        return builder;
    }

    di::ContainerBuilder installRadianceCollectorApp()
    {
        di::ContainerBuilder builder;

        addSceneItem<Sun>(builder);
        addSceneItem<VDBCloud>(builder);
        addSceneItem<CloudPTRenderer>(builder);
        addSceneItem<RadianceCollector>(builder);
        addSceneItem<Camera>(builder);

        return builder;
    }

    di::ContainerBuilder installDataset(const std::string& databasePath)
    {
        di::ContainerBuilder builder;

        builder.registerInstance(std::make_shared<Dataset::Settings>(databasePath));
        builder.registerType<Dataset>().singleInstance();

        return builder;
    }

    di::ContainerBuilder BindSceneDescription(SceneDescription& scene)
    {
        di::ContainerBuilder builder;

        builder.registerInstance(std::make_shared<Cloud::Model>(scene.cloud.model));
        builder.registerInstance(std::make_shared<Cloud::Rendering>(scene.cloud.rendering));
        builder.registerInstance(std::make_shared<Cloud>(scene.cloud));

        builder.registerInstance(std::make_shared<DirectionalLight>(scene.light));

        builder.registerInstance(std::make_shared<SceneDescription>(scene));

        return builder;
    }

    di::ContainerBuilder installFramework(
        const std::string& cloudPath, int32_t sceneId,
        uint32_t width, uint32_t height)
    {
        auto scene = SceneDescription
        {
            Cloud
            {
                Cloud::Rendering
                {
                    Cloud::Rendering::SampleStep{1.0f / 512.f},
                    Cloud::Rendering::Mode::SunMultipleScatter
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
                Color{optix::make_float3(1, 1, 1)},
                1e6
            }
        };

        di::ContainerBuilder builder;

        builder.addRegistrations(BindSceneDescription(scene));
        builder.registerInstance(std::make_shared<Camera::Settings>(width, height));

        builder.registerInstance(std::make_shared<optix::Context>(optix::Context::create()));

        builder.registerType<Resources>().singleInstance();
        builder.registerType<Scene>().singleInstance();
        builder.registerInstance(std::make_shared<BatchSettings>(sceneId * 2048, 2048));

        return builder;
    }
}
