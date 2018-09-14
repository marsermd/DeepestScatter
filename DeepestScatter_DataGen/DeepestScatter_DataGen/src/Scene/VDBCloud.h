#pragma once

#include <optixu/optixpp_namespace.h>

#include "Scene.h"
#include "SceneItem.h"
#include "Util/Resources.h"

namespace DeepestScatter
{
    class VDBCloud: public SceneItem
    {
    public:
        VDBCloud(optix::Context context, std::shared_ptr<Resources>& resources):
            context(context),
            resources(resources) {}

        virtual ~VDBCloud() override = default;

        void setCloudPath(const std::string& path);

        void Init() override;
        void Update() override {} // Does nothing.

        template <class T>
        void SetupVariables(optix::Handle<T>& scope) const;

        optix::TextureSampler   densitySampler;
    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;

        std::string resourcePath;

        optix::Buffer           densityBuffer;

        optix::Buffer           inScatterBuffer;
        optix::TextureSampler   inScatterSampler;

        void InitVolume();
        void InitInScatter();

        template <class T>
        void SetupVolumeVariables(optix::Handle<T>& scope) const;
        template <class T>
        void SetupInScatterVariables(optix::Handle<T>& scope) const;

        optix::TextureSampler createSamplerForBuffer3D(const optix::Buffer& buffer);
    };
}

