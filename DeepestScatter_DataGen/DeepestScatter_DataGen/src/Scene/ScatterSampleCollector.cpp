#include "ScatterSampleCollector.h"
#include "Util/BufferBind.h"

#pragma warning(push, 0)
#include "SceneSetup.pb.h"
#include "ScatterSample.pb.h"
#pragma warning(pop)

namespace DeepestScatter 
{
    void ScatterSampleCollector::init()
    {
        directionBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, settings.batchSize);
        positionBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, settings.batchSize);

        resetProgram = resources->loadProgram("pointGeneratorCamera.cu", "clear");
        generateProgram = resources->loadProgram("pointGeneratorCamera.cu", "generatePoints");

        setupVariables(resetProgram);
        setupVariables(generateProgram);

        reset();
        collect();
    }

    void ScatterSampleCollector::reset()
    {
        context->setRayGenerationProgram(0, resetProgram);
        context->launch(0, settings.batchSize);
    }

    void ScatterSampleCollector::collect()
    {
        std::cout << "Generating samples..." << std::endl;
        context->setRayGenerationProgram(0, generateProgram);
        context->launch(0, settings.batchSize);

        {
            BufferBind<optix::float3> directions(directionBuffer);
            BufferBind<optix::float3> positions(positionBuffer);

            std::vector<Storage::ScatterSample> samples(settings.batchSize);
            std::cout << "Serializing samples..." << std::endl;
            for(int i = 0; i < settings.batchSize; i++)
            {
                samples[i].mutable_point()->set_x(positions[i].x);
                samples[i].mutable_point()->set_y(positions[i].y);
                samples[i].mutable_point()->set_z(positions[i].z);

                samples[i].mutable_view_direction()->set_x(directions[i].x);
                samples[i].mutable_view_direction()->set_y(directions[i].y);
                samples[i].mutable_view_direction()->set_z(directions[i].z);
            }

            std::cout << "Writing samples..." << std::endl;
            dataset->batchAppend(gsl::make_span(samples), settings.batchStartId);
            std::cout << "Finished writing samples." << std::endl;
        }
    }

    template <class T>
    void ScatterSampleCollector::setupVariables(optix::Handle<T>& scope) const
    {
        scope["directionBuffer"]->setBuffer(directionBuffer);
        scope["positionBuffer"]->setBuffer(positionBuffer);
    }
}
