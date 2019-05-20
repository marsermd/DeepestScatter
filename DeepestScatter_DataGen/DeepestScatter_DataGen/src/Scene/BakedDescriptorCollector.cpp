#include "BakedDescriptorCollector.h"

#include "CUDA/DisneyDescriptor.h"
#include "Util/BufferBind.h"
#include "Util/Resources.h"

#include <string>
#include "ScatterSample.pb.h"
#include "BakedDescriptor.pb.h"
#include "CUDA/LightProbe.h"
#include "BakedInterpolationSet.pb.h"

namespace DeepestScatter
{

    void BakedDescriptorCollector::init()
    {
        positionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, settings.batchSize);
        directionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, settings.batchSize);

        bakedInterpolationsBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, settings.batchSize);
        bakedInterpolationsBuffer->setElementSize(sizeof(Gpu::BakedInterpolationSet));

        {
            BufferBind<optix::float3> positions(positionBuffer);
            BufferBind<optix::float3> directions(directionBuffer);

            for (int i = 0; i < settings.batchSize; i++)
            {
                auto sample = dataset->getRecord<Persistance::ScatterSample>(settings.batchStartId + i);

                const optix::float3 originalPosition = optix::make_float3
                (
                    sample.point().x(),
                    sample.point().y(),
                    sample.point().z()
                );

                positions[i] = originalPosition;

                // Baked descriptor doesn't know where the light shoots from.
                directions[i] = optix::make_float3(0, 0, 1);
            }
        }

        resetProgram = resources->loadProgram("lightProbeCollector.cu", "clear");
        collectProgram = resources->loadProgram("lightProbeCollector.cu", "collect");

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
        scope["interpolationSets"]->setBuffer(bakedInterpolationsBuffer);
    }

    inline Persistance::BakedDescriptor serializeProbe(Gpu::BakedInterpolationSet::Probe& probe)
    {
        constexpr auto descriptorSize = sizeof(Gpu::DisneyDescriptor);

        Persistance::BakedDescriptor res;
        std::array<uint8_t, descriptorSize> bytes{};
        memcpy(&bytes[0], &probe.descriptor, descriptorSize);

        res.set_grid(static_cast<const void*>(&bytes[0]), bytes.size() * sizeof(uint8_t));
        {
            res.mutable_position()->set_x(probe.position.x);
            res.mutable_position()->set_y(probe.position.y);
            res.mutable_position()->set_z(probe.position.z);
        }
        {
            res.mutable_direction()->set_x(probe.direction.x);
            res.mutable_direction()->set_y(probe.direction.y);
            res.mutable_direction()->set_z(probe.direction.z);
        }
        {
            res.set_power(probe.power);
        }
        return res;
    }

    void BakedDescriptorCollector::recordToDataset() const
    {
        BufferBind<Gpu::BakedInterpolationSet> descriptors(bakedInterpolationsBuffer);
        BufferBind<optix::float3> positions(positionBuffer);
        BufferBind<optix::float3> directions(directionBuffer);

        std::vector<Persistance::BakedInterpolationSet> serializedInterpolations(settings.batchSize);
        for (int i = 0; i < settings.batchSize; i++)
        {

            Gpu::BakedInterpolationSet& interpolationSet = descriptors[i];

            serializedInterpolations[i].mutable_a()->CopyFrom(serializeProbe(interpolationSet.a));
            serializedInterpolations[i].mutable_b()->CopyFrom(serializeProbe(interpolationSet.b));
            serializedInterpolations[i].mutable_c()->CopyFrom(serializeProbe(interpolationSet.c));
            serializedInterpolations[i].mutable_d()->CopyFrom(serializeProbe(interpolationSet.d));
        }

        std::cout << "Writing interpolation sets..." << std::endl;
        dataset->batchAppend(gsl::make_span(serializedInterpolations), settings.batchStartId);
        std::cout << "Finished writing interpolation sets." << std::endl;
    }
}
