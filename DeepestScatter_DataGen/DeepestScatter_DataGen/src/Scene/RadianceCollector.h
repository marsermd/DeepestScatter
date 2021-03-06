#pragma once

#include <optixu/optixpp_namespace.h>
#include <utility>
#include <memory>

#include "Scene/SceneItem.h"
#include "Util/Resources.h"
#include "Util/Dataset/Dataset.h"
#include "Util/Dataset/BatchSettings.h"
#include "CUDA/PointRadianceTask.h"

namespace DeepestScatter
{
    class Resources;

    class RadianceCollector : public SceneItem
    {
    public:
        RadianceCollector(
            std::shared_ptr<optix::Context> context, 
            std::shared_ptr<Resources> resources,
            std::shared_ptr<Dataset> dataset,
            std::shared_ptr<BatchSettings> settings):
            context(*context.get()),
            resources(std::move(resources)),
            dataset(std::move(dataset)),
            settings(*settings.get())
        {}

        void init() override;
        void reset() override;
        void update() override;
        bool isCompleted() override;

        int32_t getConvergedCount() const;
        int32_t getRemainingCount() const;
    private:
        void setupVariables(optix::Program& handle);
        void scheduleTasks(const gsl::span<Gpu::PointRadianceTask>& tasks);
        void recordToDataset();


        optix::Context context;
        std::shared_ptr<Resources> resources;
        std::shared_ptr<Dataset> dataset;

        BatchSettings settings;


        bool allPixelsConverged = false;
        uint32_t frameId = 0;
        uint32_t threadsCount;
        uint32_t taskRepeatCount;

        optix::Program resetProgram;
        optix::Program renderProgram;

        optix::Buffer tasksBuffer;

        std::vector<Gpu::PointRadianceTask> convergedTasks;
    };
}
