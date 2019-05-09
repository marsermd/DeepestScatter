#include "Tasks.h"
#include "Util/Dataset/Dataset.h"
#include "installers.h"
#include "DisneyDescriptor.pb.h"
#include "Scene/RadianceCollector.h"
#include "Scene/DisneyDescriptorCollector.h"
#include "BakedDescriptor.pb.h"
#include "Scene/BakedDescriptorCollector.h"
#include "Scene/Cameras/Camera.h"
#include "Scene/Cameras/DisneyRenderer.h"

namespace DeepestScatter
{
    namespace di = Hypodermic;

    uint32_t width = 640u;
    uint32_t height = 512u;

    std::queue<GuiExecutionLoop::LazyTask> Tasks::renderCloud(const std::string &cloudPath, float sizeM)
    {
        std::queue<GuiExecutionLoop::LazyTask> tasks;

        GuiExecutionLoop::LazyTask task = [=]()
        {
            di::ContainerBuilder taskBuilder;

            Persistance::SceneSetup sceneSetup{};
            sceneSetup.set_cloud_path(cloudPath);
            sceneSetup.set_cloud_size_m(sizeM);
            sceneSetup.mutable_light_direction()->set_x(-0.586f);
            sceneSetup.mutable_light_direction()->set_y(-0.766f);
            sceneSetup.mutable_light_direction()->set_z(-0.271f);
            //sceneSetup.mutable_light_direction()->set_x(0.8f);
            //sceneSetup.mutable_light_direction()->set_y(-0.25f);
            //sceneSetup.mutable_light_direction()->set_z(0.11f);

            taskBuilder.addRegistrations(installFramework(width, height));
            taskBuilder.addRegistrations(installSceneSetup(sceneSetup, ".", Cloud::Rendering::Mode::SunAndSkyAllScatter, Cloud::Model::Mipmaps::On));
            taskBuilder.addRegistrations(installApp());
            taskBuilder.registerType<DisneyRenderer>().as<ARenderer>().singleInstance();

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

                taskBuilder.addRegistrations(installFramework(width, height));
                taskBuilder.addRegistrations(installSceneSetup(
                    sceneSetup, cloudRoot, Cloud::Rendering::Mode::SunMultipleScatter, Cloud::Model::Mipmaps::On));
                taskBuilder.addRegistrations(installApp());
                taskBuilder.registerInstance(std::make_shared<BatchSettings>(i * 2048, 2048));
                taskBuilder.addRegistrations(collector);

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
    void Tasks::addCollector<Persistance::BakedDescriptor>(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<BakedDescriptorCollector>().as<SceneItem>().asSelf().singleInstance();
    }
}
