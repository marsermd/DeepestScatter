#pragma once

#include <optixu/optixpp_namespace.h>
#include <utility>
#include <memory>

#include "Scene/SceneItem.h"
#include "Util/Dataset/Dataset.h"
#include "Util/Dataset/BatchSettings.h"
#include "VDBCloud.h"

namespace DeepestScatter
{
    class Resources;

    class BakedDescriptorCollector : public SceneItem
    {
    public:
        BakedDescriptorCollector(
            std::shared_ptr<optix::Context> context,
            std::shared_ptr<Resources> resources,
            std::shared_ptr<Dataset> dataset,
            std::shared_ptr<BatchSettings> settings,
            std::shared_ptr<VDBCloud> cloud,
            std::shared_ptr<Cloud::Model> cloudModel) :
            context(*context.get()),
            resources(std::move(resources)),
            dataset(std::move(dataset)),
            settings(*settings.get()),
            cloud(cloud)
        {
            cloud->disableRendering();
        }

        void init() override;
        void reset() override;
        void update() override {};

    private:
        void collect();
        void setupVariables(optix::Program& scope);
        void recordToDataset() const;

        optix::Context context;
        std::shared_ptr<Resources> resources;
        std::shared_ptr<Dataset> dataset;
        std::shared_ptr<VDBCloud> cloud;

        BatchSettings settings;

        optix::Program resetProgram;
        optix::Program collectProgram;

        optix::Buffer positionBuffer;
        optix::Buffer directionBuffer;
        optix::Buffer descriptorsBuffer;
    };
}
