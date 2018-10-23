#pragma once

#include <optixu/optixpp_namespace.h>

#include "SceneItem.h"
#include "SceneDescription.h"
#include "Util/Resources.h"

namespace DeepestScatter
{
    class VDBCloud: public SceneItem
    {
    public:
        typedef Cloud::Model Settings;

        VDBCloud(std::shared_ptr<Settings> settings, std::shared_ptr<optix::Context> context, std::shared_ptr<Resources> resources);

        void init() override;
        void reset() override {}
        void update() override {} // Does nothing.

        template <class T>
        void SetupVariables(optix::Handle<T>& scope) const;

        optix::TextureSampler   densitySampler;
    private:
        optix::Context context;
        std::shared_ptr<Resources> resources;
        Settings settings;

        optix::Buffer           densityBuffer;
        optix::float3           bboxSize;
        float                   textureScale;
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

