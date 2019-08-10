#include "Tasks.h"

#include <filesystem>

#include "Util/Dataset/Dataset.h"
#include "installers.h"
#include "DisneyDescriptor.pb.h"
#include "Scene/RadianceCollector.h"
#include "Scene/DisneyDescriptorCollector.h"
#include "Scene/BakedDescriptorCollector.h"
#include "Scene/Cameras/Camera.h"
#include "Scene/Cameras/DisneyRenderer.h"
#include "Scene/Cameras/BakedRenderer.h"
#include "Scene/Cameras/EmptyRenderer.h"
#include "BakedInterpolationSet.pb.h"
#include "Scene/Cameras/PathTracingRenderer.h"
#include "ScatterSample.pb.h"
#include "Scene/ScatterSampleCollector.h"

namespace DeepestScatter
{
    namespace di = Hypodermic;

    uint32_t width = 1792u;
    uint32_t height = 1024u;

    enum class LightDirection {
        Front, Back, Side
    };

    optix::float3 getLightDirection(LightDirection direction)
    {
        switch (direction) {
        case LightDirection::Front:
            return optix::make_float3(-0.586f, -0.766f, -0.271f);
        case LightDirection::Side:
            return optix::make_float3(-0.03f, -0.25f, 0.8f);
        case LightDirection::Back:
            return optix::make_float3(0.586f, -0.766f, -0.271f);
        default:
            throw std::exception("Unexpected direction");
        }

    }

    std::queue<GuiExecutionLoop::LazyTask> Tasks::renderCloud(const std::string &cloudPath, float sizeM)
    {
        std::queue<GuiExecutionLoop::LazyTask> tasks;

        GuiExecutionLoop::LazyTask task = [=]()
        {
            di::ContainerBuilder taskBuilder;

            Persistance::SceneSetup sceneSetup{};
            sceneSetup.set_cloud_path(cloudPath);
            sceneSetup.set_cloud_size_m(sizeM);

            optix::float3 direction = getLightDirection(LightDirection::Front);
            sceneSetup.mutable_light_direction()->set_x(direction.x);
            sceneSetup.mutable_light_direction()->set_y(direction.y);
            sceneSetup.mutable_light_direction()->set_z(direction.z);

            //sceneSetup.mutable_light_direction()->set_x(0.8f);
            //sceneSetup.mutable_light_direction()->set_y(-0.25f);
            //sceneSetup.mutable_light_direction()->set_z(0.11f);

            using TRenderer = BakedRenderer;
            taskBuilder.registerType<TRenderer>().as<ARenderer>().singleInstance();
            auto outputPath = 
                std::filesystem::path("../../Data/Renders") / 
                std::filesystem::path(cloudPath).filename().replace_extension(TRenderer::NAME + ".exr");
            taskBuilder.addRegistrations(installFramework(width, height, outputPath));
            taskBuilder.addRegistrations(installSceneSetup(sceneSetup, ".", Cloud::Rendering::Mode::SunAndSkyAllScatter, Cloud::Model::Mipmaps::On));
            taskBuilder.addRegistrations(installApp());

            auto container = taskBuilder.build();

            auto camera = container->resolve<Camera>();
            camera->completed = false;

            return container;
        };

        tasks.push(task);

        return tasks;
    }

    std::queue<DeepestScatter::GuiExecutionLoop::LazyTask> Tasks::collect(
        const std::string& cloudRoot,
        di::ContainerBuilder collector,
        std::shared_ptr<Dataset>& dataset,
        int32_t startSceneId,
        std::shared_ptr<di::Container> rootContainer)
    {
        std::queue<GuiExecutionLoop::LazyTask> tasks;

        const size_t sceneCount = dataset->getRecordsCount<Persistance::SceneSetup>();

        for (int32_t i = startSceneId; i < sceneCount; i++)
        {
            const auto sceneSetup = dataset->getRecord<Persistance::SceneSetup>(i);

            GuiExecutionLoop::LazyTask task = [=]()
            {
                di::ContainerBuilder taskBuilder;

                taskBuilder.addRegistrations(installFramework(width, height, std::filesystem::path()));
                taskBuilder.addRegistrations(installSceneSetup(
                    sceneSetup, cloudRoot, Cloud::Rendering::Mode::SunMultipleScatter, Cloud::Model::Mipmaps::On));
                taskBuilder.addRegistrations(installApp());
                taskBuilder.registerInstance(std::make_shared<BatchSettings>(i * 2048, 2048));
                taskBuilder.addRegistrations(collector);
                taskBuilder.registerType<EmptyRenderer>().as<ARenderer>().singleInstance();

                auto container = taskBuilder.buildNestedContainerFrom(*rootContainer.get());

                //auto camera = container->resolve<Camera>();
                //camera->completed = false;

                return container;
            };

            tasks.push(task);
        }

        return tasks;
    }


    template<>
    void Tasks::addCollector<Persistance::ScatterSample>(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<ScatterSampleCollector>().as<SceneItem>().asSelf().singleInstance();
    }

    template<>
    void Tasks::addCollector<Persistance::Result>(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<RadianceCollector>().as<SceneItem>().asSelf().singleInstance();
    }

    template<>
    void Tasks::addCollector<Persistance::DisneyDescriptor>(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<DisneyDescriptorCollector>().as<SceneItem>().asSelf().singleInstance();
    }

    template<>
    void Tasks::addCollector<Persistance::BakedInterpolationSet>(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<BakedDescriptorCollector>().as<SceneItem>().asSelf().singleInstance();
    }
}
