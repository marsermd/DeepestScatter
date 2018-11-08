#include "Installers.h"

#include "Scene/SceneDescription.h"

#include <memory>

#include "Scene/Sun.h"
#include "Scene/VDBCloud.h"
#include "Scene/CloudPTRenderer.h"
#include "Scene/ScatterSampleCollector.h"
#include "Scene/Camera.h"

#include "Scene/Scene.h"
#include "Scene/RadianceCollector.h"
#include <filesystem>
#include "Scene/DisneyDescriptorCollector.h"

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

    di::ContainerBuilder installSampleCollectorApp()
    {
        di::ContainerBuilder builder;

        addSceneItem<Sun>(builder);
        addSceneItem<VDBCloud>(builder);
        addSceneItem<CloudPTRenderer>(builder);
        addSceneItem<ScatterSampleCollector>(builder);
        addSceneItem<Camera>(builder);

        return builder;
    }

    di::ContainerBuilder installDisneyDescriptorCollectorApp()
    {
        di::ContainerBuilder builder;

        addSceneItem<Sun>(builder);
        addSceneItem<VDBCloud>(builder);
        addSceneItem<CloudPTRenderer>(builder);
        addSceneItem<DisneyDescriptorCollector>(builder);
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

    di::ContainerBuilder bindSceneDescription(SceneDescription& scene)
    {
        di::ContainerBuilder builder;

        builder.registerInstance(std::make_shared<Cloud::Model>(scene.cloud.model));
        builder.registerInstance(std::make_shared<Cloud::Rendering>(scene.cloud.rendering));
        builder.registerInstance(std::make_shared<Cloud>(scene.cloud));

        builder.registerInstance(std::make_shared<DirectionalLight>(scene.light));

        builder.registerInstance(std::make_shared<SceneDescription>(scene));

        return builder;
    }

    Hypodermic::ContainerBuilder installSceneSetup(const Storage::SceneSetup& sceneSetup, const std::string& cloudsRoot)
    {
        std::filesystem::path cloudPath(cloudsRoot);
        cloudPath /= sceneSetup.cloud_path();
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
                    cloudPath.string(),
                    Cloud::Model::Mipmaps::Off,
                    Cloud::Model::Size{Meter{sceneSetup.cloud_size_m()}}
                }
            },
            DirectionalLight
            {
                optix::make_float3(
                    sceneSetup.light_direction().x(), 
                    sceneSetup.light_direction().y(),
                    sceneSetup.light_direction().z()
                ),
                Color{optix::make_float3(1, 1, 1)},
                1e6
            }
        };

        return bindSceneDescription(scene);
    }

    di::ContainerBuilder installFramework(int32_t sceneId, uint32_t width, uint32_t height)
    {
        di::ContainerBuilder builder;

        builder.registerInstance(std::make_shared<Camera::Settings>(width, height));

        builder.registerInstance(std::make_shared<optix::Context>(optix::Context::create()));

        builder.registerType<Resources>().singleInstance();
        builder.registerType<Scene>().singleInstance();
        builder.registerInstance(std::make_shared<BatchSettings>(sceneId * 2048, 2048));

        return builder;
    }
}
