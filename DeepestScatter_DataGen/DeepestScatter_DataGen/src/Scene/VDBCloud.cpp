#include "VDBCloud.h"

#include <iostream>
#include <gsl/gsl>

namespace DeepestScatter
{
    VDBCloud::VDBCloud(std::shared_ptr<Settings> settings, std::shared_ptr<optix::Context> context, std::shared_ptr<Resources> resources) :
        settings(*settings.get()),
        context(*context.get()),
        resources(resources)
    {
    }

    void VDBCloud::init()
    {
        InitVolume();
        InitInScatter();
        setupVariables(context);
    }

    void VDBCloud::disableRendering()
    {
        std::cout << "NOTE: RENDERING IS NOT SUPPORTED!" << std::endl;
        isRenderingEnabled = false;
    }

    optix::size_t3 VDBCloud::getResolution() const
    {
        RTsize sizeX, sizeY, sizeZ;
        densityBuffer->getSize(sizeX, sizeY, sizeZ);
        return optix::make_size_t3(sizeX, sizeY, sizeZ);
    }

    float VDBCloud::getVoxelSizeInMeters() const
    {
        optix::size_t3 size = getResolution();
        size_t max = std::max({size.x, size.y, size.z});

        return settings.size / max;
    }

    float VDBCloud::getVoxelSizeInTermsOfFreePath() const
    {
        return getVoxelSizeInMeters() / settings.meanFreePath;
    }

    void VDBCloud::InitVolume()
    {
        auto cloud = resources->loadVolumeBuffer(settings.vdbPath, static_cast<bool>(settings.mipmapsOn));
        densityBuffer = std::get<optix::Buffer>(cloud);
        bboxSize = std::get<optix::float3>(cloud);

        densitySampler = createSamplerForBuffer3D(densityBuffer);
    }

    void VDBCloud::InitInScatter()
    {
        if (!isRenderingEnabled)
        {
            inScatterBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 1, 1, 1);
            inScatterSampler = createSamplerForBuffer3D(inScatterBuffer);
            return;
        }
        RTsize sizeX, sizeY, sizeZ;
        densityBuffer->getSize(sizeX, sizeY, sizeZ);

        // RT_BUFFER_INPUT only means that it resides on GPU.
        // Don't be fooled by it's name. We are going to write in the inScatterBuffer during inScatter program.
        inScatterBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, sizeX, sizeY, sizeZ);

        optix::Program inScatter = resources->loadProgram("inScatter.cu", "inScatter");
        setupVolumeVariables(inScatter);

        inScatter["inScatterBuffer"]->setBuffer(inScatterBuffer);
        context->setRayGenerationProgram(0, inScatter);

        context->validate();
        context->launch(0, sizeX, sizeY, sizeZ);

        // We have to call destroy explicitly, because inScatter program holds inScatterBuffer as a buffer.
        // And using a buffer as a sampler and a buffer simultaneously is invalid.
        inScatter->destroy();

        inScatterSampler = createSamplerForBuffer3D(inScatterBuffer);
    }

    template <class T>
    void VDBCloud::setupVariables(optix::Handle<T>& scope) const
    {
        scope["voxelSizeInTermsOfFreePath"]->setFloat(getVoxelSizeInTermsOfFreePath());
        scope["voxelSizeInMeters"]->setFloat(getVoxelSizeInMeters());

        setupVolumeVariables(scope);
        setupInScatterVariables(scope);
    }

    template<class T>
    void VDBCloud::setupVolumeVariables(optix::Handle<T>& scope) const
    {
        float maxSize = std::max({ bboxSize.x, bboxSize.y, bboxSize .z });
        RTsize textureX, textureY, textureZ;
        densityBuffer->getSize(textureX, textureY, textureZ);

        scope["bboxSize"]->setFloat(bboxSize.x / maxSize, bboxSize.y / maxSize, bboxSize.z / maxSize);
        scope["textureScale"]->setFloat(maxSize / textureX, maxSize / textureY, maxSize / textureZ);
        scope["densityTextureId"]->setInt(densitySampler->getId());
        scope["density"]->setTextureSampler(densitySampler);
        scope["densityMultiplier"]->setFloat(settings.size / settings.meanFreePath);
        scope["cloudSizeInMeters"]->setFloat(settings.size);
    }

    template<class T>
    void VDBCloud::setupInScatterVariables(optix::Handle<T>& scope) const
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
            RT_FILTER_LINEAR
        );

        sampler3D->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler3D->setBuffer(buffer);
        return sampler3D;
    }
}
