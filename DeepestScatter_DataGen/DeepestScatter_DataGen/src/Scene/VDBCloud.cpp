#include "VDBCloud.h"

#include <iostream>
#include <gsl/gsl>

namespace DeepestScatter
{
    void VDBCloud::setCloudPath(const std::string & path)
    {
        resourcePath = path;
    }

    void VDBCloud::Init()
    {
        InitVolume();
        InitInScatter();
        SetupVariables(context);
    }

    void VDBCloud::InitVolume()
    {
        densityBuffer = resources->loadVolumeBuffer(resourcePath);
        densitySampler = createSamplerForBuffer3D(densityBuffer);
    }

    void VDBCloud::InitInScatter()
    {
        RTsize sizeX, sizeY, sizeZ;
        densityBuffer->getSize(sizeX, sizeY, sizeZ);

        // Here RT_BUFFER_INPUT only means that it resides on GPU.
        // Don't be fooled by that. We are going to write in the inScatterBuffer during inScatter program.
        inScatterBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, sizeX, sizeY, sizeZ);

        optix::Program inScatter = resources->loadProgram("inScatter.cu", "inScatter");
        SetupVolumeVariables(inScatter);

        inScatter["resultBuffer"]->setBuffer(inScatterBuffer);
        context->setRayGenerationProgram(0, inScatter);

        context->validate();
        context->launch(0, sizeX, sizeY, sizeZ);

        // We have to call destriy explicitly, because inScatter program holds inScatterBuffer as a buffer.
        // Using a buffer as a sampler and a buffer simultaniously is invalid.
        inScatter->destroy();

        inScatterSampler = createSamplerForBuffer3D(inScatterBuffer);
    }

    template <class T>
    void VDBCloud::SetupVariables(optix::Handle<T>& scope) const
    {
        SetupVolumeVariables(scope);
        SetupInScatterVariables(scope);
    }

    template<class T>
    void VDBCloud::SetupVolumeVariables(optix::Handle<T>& scope) const
    {
        RTsize sizeX, sizeY, sizeZ;
        densityBuffer->getSize(sizeX, sizeY, sizeZ);
        float maxSize = gsl::narrow<float>(std::max({ sizeX, sizeY, sizeZ }));

        scope["boxSize"]->setFloat(sizeX / maxSize, sizeY / maxSize, sizeZ / maxSize);
        scope["cloud"]->setTextureSampler(densitySampler);
    }

    template<class T>
    void VDBCloud::SetupInScatterVariables(optix::Handle<T>& scope) const
    {
        scope["inScatter"]->setTextureSampler(inScatterSampler);
    }

    optix::TextureSampler VDBCloud::createSamplerForBuffer3D(const optix::Buffer& buffer)
    {
        optix::TextureSampler sampler3D = context->createTextureSampler();

        for (uint32_t dim = 0; dim < 3; dim++)
        {
            sampler3D->setWrapMode(dim, RT_WRAP_CLAMP_TO_EDGE);
        }

        sampler3D->setFilteringModes(
            RT_FILTER_LINEAR,
            RT_FILTER_LINEAR,
            RT_FILTER_NONE
        );

        sampler3D->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler3D->setBuffer(buffer);
        return sampler3D;
    }
}
