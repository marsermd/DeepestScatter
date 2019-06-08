#pragma once

#include <memory>

#include <optixu/optixpp_namespace.h>

#include "Scene/SceneItem.h"
#include "Util/Resources.h"
#include "Util/Dataset/Dataset.h"
#include "VDBCloud.h"
#include "Util/Dataset/BatchSettings.h"

namespace DeepestScatter
{
    class ScatterSampleCollector: public SceneItem
    {
    public:

        ScatterSampleCollector(
            std::shared_ptr<optix::Context> context, 
            std::shared_ptr<Resources> resources, 
            std::shared_ptr<Dataset> dataset,
            std::shared_ptr<BatchSettings> settings, 
            std::shared_ptr<SceneDescription> sceneDescription,
            std::shared_ptr<VDBCloud> cloud) :
            context(*context.get()),
            resources(std::move(resources)),
            dataset(std::move(dataset)),
            settings(*settings.get()),
            sceneDescription(*sceneDescription.get())
        {
            cloud->disableRendering();
        }

        void collect();

        void init() override;
        void reset() override;
        void update() override;

    private:
        template <class T>
        void setupVariables(optix::Handle<T>& scope) const;

        optix::Context context;
        std::shared_ptr<Resources> resources;
        std::shared_ptr<Dataset> dataset;
        std::shared_ptr<VDBCloud> cloud;

        BatchSettings settings;
        SceneDescription sceneDescription;

        optix::Program resetProgram;
        optix::Program generateProgram;

        optix::Buffer directionBuffer;
        optix::Buffer positionBuffer;
    };
}
