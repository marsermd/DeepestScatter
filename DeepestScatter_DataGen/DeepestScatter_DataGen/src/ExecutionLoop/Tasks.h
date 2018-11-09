#pragma once
#include "Hypodermic/Hypodermic.h"
#include "GuiExecutionLoop.h"
#include "Scene/RadianceCollector.h"
#include "Result.pb.h"

namespace Storage
{
    class Result;
}

namespace DeepestScatter
{
    class Tasks
    {
    public:
        enum class CollectMode
        {
            Reset,
            Continue
        };

        static std::queue<GuiExecutionLoop::LazyTask> renderCloud(const std::string &cloudPath, float sizeM);

        template<typename T>
        static std::queue<GuiExecutionLoop::LazyTask> collect(
            const std::string& databasePath,
            const std::string& cloudRoot,
            CollectMode mode);
    private:
        static std::queue<GuiExecutionLoop::LazyTask> collect(
            const std::string& cloudRoot,
            Hypodermic::ContainerBuilder collector,
            std::shared_ptr<Dataset>& dataset,
            int32_t startSceneId,
            std::shared_ptr<Hypodermic::Container> rootContainer);

        template<typename T>
        static void addCollector(Hypodermic::ContainerBuilder& builder);
    };

    template <typename T>
    std::queue<GuiExecutionLoop::LazyTask> Tasks::collect(
        const std::string& databasePath,
        const std::string& cloudRoot,
        CollectMode mode)
    {
        namespace di = Hypodermic;

        di::ContainerBuilder datasetBuilder;
        datasetBuilder.addRegistrations(installDataset(databasePath));
        auto rootContainer = datasetBuilder.build();

        auto dataset = rootContainer->resolve<Dataset>();

        di::ContainerBuilder collector;
        addCollector<T>(collector);

        int32_t startSceneId = 0;
        if (mode == CollectMode::Reset)
        {
            dataset->dropTable<T>();
            startSceneId = 0;
        }
        else
        {
            startSceneId = dataset->getRecordsCount<T>() / 2048;
        }

        return collect(cloudRoot, collector, dataset, startSceneId, rootContainer);
    }
}
