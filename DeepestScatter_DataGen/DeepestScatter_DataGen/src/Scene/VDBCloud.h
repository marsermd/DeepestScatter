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
        void setupVariables(optix::Handle<T>& scope) const;

        void disableRendering();

        optix::size_t3 getResolution();
        float getVoxelSize();
        float getVoxelSizeInTermsOfFreePath();

    private:
        Settings settings;
        optix::Context context;
        std::shared_ptr<Resources> resources;

        optix::Buffer           densityBuffer;
        optix::TextureSampler   densitySampler;
        optix::float3           bboxSize;
        float                   textureScale;
        optix::Buffer           inScatterBuffer;
        optix::TextureSampler   inScatterSampler;

        bool isRenderingEnabled = true;

        void InitVolume();
        void InitInScatter();

        template <class T>
        void setupVolumeVariables(optix::Handle<T>& scope) const;
        template <class T>
        void setupInScatterVariables(optix::Handle<T>& scope) const;

        optix::TextureSampler createSamplerForBuffer3D(const optix::Buffer& buffer);
    };
}

