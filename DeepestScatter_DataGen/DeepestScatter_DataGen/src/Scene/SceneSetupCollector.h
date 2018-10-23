#pragma once

#include <memory>

#include <optixu/optixpp_namespace.h>

#include "Scene/SceneItem.h"
#include "Util/Resources.h"
#include "Util/Dataset/Dataset.h"
#include "VDBCloud.h"

namespace DeepestScatter
{
    struct BatchSettings
    {
        explicit BatchSettings(uint32_t size)
            : size(size)
        {
        }

        uint32_t size;
    };

    class SceneSetupCollector: public SceneItem
    {
    public:

        SceneSetupCollector(std::shared_ptr<optix::Context> context, std::shared_ptr<Resources> resources, std::shared_ptr<Dataset> dataset,
            std::shared_ptr<BatchSettings> settings, std::shared_ptr<SceneDescription> sceneDescription) :
            context(*context.get()),
            resources(std::move(resources)),
            dataset(std::move(dataset)),
            settings(*settings.get()),
            sceneDescription(*sceneDescription.get())
        {
        }

        void collect();

        void init() override;
        void reset() override;
        void update() override {};

    private:
        template <class T>
        void SetupVariables(optix::Handle<T>& scope) const;

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

    template <class T>
    void SceneSetupCollector::SetupVariables(optix::Handle<T>& scope) const
    {
        scope["directionBuffer"]->setBuffer(directionBuffer);
        scope["positionBuffer"]->setBuffer(positionBuffer);
    }
}
