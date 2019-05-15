#include "BakedDescriptorCollector.h"

#include "CUDA/DisneyDescriptor.h"
#include "Util/BufferBind.h"
#include "Util/Resources.h"

#include <string>
#include "ScatterSample.pb.h"
#include "BakedDescriptor.pb.h"
#include "SceneSetup.pb.h"
#include "CUDA/LightProbe.h"

namespace DeepestScatter
{
    float roundToStep(float x, float step)
    {
        return std::round(x / step) * step;
    }

    optix::float3 roundToStep(optix::float3 v, float step)
    {
        return optix::make_float3(
            roundToStep(v.x, step),
            roundToStep(v.y, step),
            roundToStep(v.z, step)
        );
    }

    void BakedDescriptorCollector::init()
    {
        positionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, settings.batchSize);
        directionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, settings.batchSize);

        descriptorsBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, settings.batchSize);
        descriptorsBuffer->setElementSize(sizeof(Gpu::DisneyDescriptor));

        {
            BufferBind<optix::float3> positions(positionBuffer);
            BufferBind<optix::float3> directions(directionBuffer);

            // 1.0f / Gpu::LightProbe::RESOLUTION because we will place the baked points in a grid of RESOLUTIONxRESOLUTIONxRESOLUTION
            const float step = 1.0 / Gpu::LightProbe::RESOLUTION;

            for (int i = 0; i < settings.batchSize; i++)
            {
                auto sample = dataset->getRecord<Persistance::ScatterSample>(settings.batchStartId + i);

                const optix::float3 originalPosition = optix::make_float3
                (
                    sample.point().x(),
                    sample.point().y(),
                    sample.point().z()
                );

                // Baked descriptor only lands at the requested point with some precision.
                positions[i] = roundToStep(originalPosition, step);

                // Baked descriptor doesn't know where the light shoots from.
                directions[i] = optix::make_float3(0, 0, 1);
            }
        }

        resetProgram = resources->loadProgram("disneyDescriptorCollector.cu", "clear");
        collectProgram = resources->loadProgram("disneyDescriptorCollector.cu", "collect");

        setupVariables(resetProgram);
        setupVariables(collectProgram);

        reset();
        collect();
    }

    void BakedDescriptorCollector::reset()
    {
        context->setRayGenerationProgram(0, resetProgram);
        context->launch(0, settings.batchSize);
    }

    void BakedDescriptorCollector::collect()
    {
        context->setRayGenerationProgram(0, collectProgram);
        context->launch(0, settings.batchSize);

        recordToDataset();
    }

    void BakedDescriptorCollector::setupVariables(optix::Program& scope)
    {
        scope["directionBuffer"]->setBuffer(directionBuffer);
        scope["positionBuffer"]->setBuffer(positionBuffer);
        scope["descriptors"]->setBuffer(descriptorsBuffer);
    }

    void BakedDescriptorCollector::recordToDataset() const
    {
        BufferBind<Gpu::DisneyDescriptor> descriptors(descriptorsBuffer);
        BufferBind<optix::float3> positions(positionBuffer);
        BufferBind<optix::float3> directions(directionBuffer);

        std::vector<Persistance::BakedDescriptor> serializedDescriptors(settings.batchSize);
        for (int i = 0; i < settings.batchSize; i++)
        {
            constexpr size_t pointsInLayer =
                Gpu::DisneyDescriptor::Layer::SIZE_Z *
                Gpu::DisneyDescriptor::Layer::SIZE_Y *
                Gpu::DisneyDescriptor::Layer::SIZE_X;
            constexpr auto bytesLength = Gpu::DisneyDescriptor::LAYERS_CNT * pointsInLayer;

            Gpu::DisneyDescriptor& descriptor = descriptors[i];

            std::array<uint8_t, bytesLength> bytes{};
            for (size_t layerId = 0; layerId < Gpu::DisneyDescriptor::LAYERS_CNT; layerId++)
            {
                memcpy(&bytes[layerId * pointsInLayer], &descriptor.layers[layerId], pointsInLayer * sizeof(uint8_t));
            }

            serializedDescriptors[i].set_grid(static_cast<const void*>(&bytes[0]), bytes.size() * sizeof(uint8_t));

            {
                optix::float3 position = positions[i];
                serializedDescriptors[i].mutable_position()->set_x(position.x);
                serializedDescriptors[i].mutable_position()->set_y(position.y);
                serializedDescriptors[i].mutable_position()->set_z(position.z);
            }
            {
                optix::float3 direction = directions[i];
                serializedDescriptors[i].mutable_direction()->set_x(direction.x);
                serializedDescriptors[i].mutable_direction()->set_y(direction.y);
                serializedDescriptors[i].mutable_direction()->set_z(direction.z);
            }
        }

        std::cout << "Writing descriptors..." << std::endl;
        dataset->batchAppend(gsl::make_span(serializedDescriptors), settings.batchStartId);
        std::cout << "Finished writing descriptors." << std::endl;
    }
}
