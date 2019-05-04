#include "DisneyDescriptorCollector.h"
#include "CUDA/DisneyDescriptor.h"
#include "Util/BufferBind.h"
#include "DisneyDescriptor.pb.h"

#include <string>
#include "ScatterSample.pb.h"

namespace DeepestScatter
{
    void DisneyDescriptorCollector::init()
    {
        positionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, settings.batchSize);
        directionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, settings.batchSize);

        descriptorsBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, settings.batchSize);
        descriptorsBuffer->setElementSize(sizeof(Gpu::DisneyDescriptor));

        {
            BufferBind<optix::float3> positions(positionBuffer);
            BufferBind<optix::float3> directions(directionBuffer);

            for (int i = 0; i < settings.batchSize; i++)
            {
                auto sample = dataset->getRecord<Persistance::ScatterSample>(settings.batchStartId + i);
                positions[i] = optix::make_float3
                (
                    sample.point().x(),
                    sample.point().y(),
                    sample.point().z()
                );

                directions[i] = optix::make_float3
                (
                    sample.view_direction().x(),
                    sample.view_direction().y(),
                    sample.view_direction().z()
                );

            }
        }

        resetProgram = resources->loadProgram("disneyDescriptorCollector.cu", "clear");
        collectProgram = resources->loadProgram("disneyDescriptorCollector.cu", "collect");

        setupVariables(resetProgram);
        setupVariables(collectProgram);

        reset();
        collect();
    }

    void DisneyDescriptorCollector::reset()
    {
        context->setRayGenerationProgram(0, resetProgram);
        context->launch(0, settings.batchSize);
    }

    void DisneyDescriptorCollector::collect()
    {
        context->setRayGenerationProgram(0, collectProgram);
        context->launch(0, settings.batchSize);

        recordToDataset();
    }

    void DisneyDescriptorCollector::setupVariables(optix::Program& scope)
    {
        scope["directionBuffer"]->setBuffer(directionBuffer);
        scope["positionBuffer"]->setBuffer(positionBuffer);
        scope["descriptors"]->setBuffer(descriptorsBuffer);
    }

    void DisneyDescriptorCollector::recordToDataset() const
    {
        BufferBind<Gpu::DisneyDescriptor> descriptors(descriptorsBuffer);

        std::vector<Persistance::DisneyDescriptor> serializedDescriptors(settings.batchSize);
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
        }

        std::cout << "Writing descriptors..." << std::endl;
        dataset->batchAppend(gsl::make_span(serializedDescriptors), settings.batchStartId);
        std::cout << "Finished writing descriptors." << std::endl;
    }
}
