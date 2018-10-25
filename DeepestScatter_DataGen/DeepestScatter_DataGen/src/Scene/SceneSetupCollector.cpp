#include "SceneSetupCollector.h"
#include "Util/BufferBind.h"

#pragma warning(push, 0)
#include "SceneSetup.pb.h"
#include "ScatterSample.pb.h"
#pragma warning(pop)

namespace DeepestScatter 
{
    void SceneSetupCollector::init()
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

    void SceneSetupCollector::reset()
    {
        context->setRayGenerationProgram(0, resetProgram);
        context->launch(0, settings.batchSize);
    }

    void SceneSetupCollector::collect()
    {
        std::cout << "Generating samples..." << std::endl;
        context->setRayGenerationProgram(0, generateProgram);
        context->launch(0, settings.batchSize);

        {
            BufferBind<optix::float3> directions(directionBuffer);
            BufferBind<optix::float3> positions(positionBuffer);

            {
                Storage::SceneSetup sceneSetup;

                sceneSetup.set_cloud_path(sceneDescription.cloud.model.vdbPath);
                sceneSetup.set_cloud_size_m(sceneDescription.cloud.model.size);
                sceneSetup.mutable_light_direction()->set_x(sceneDescription.light.direction.x);
                sceneSetup.mutable_light_direction()->set_y(sceneDescription.light.direction.y);
                sceneSetup.mutable_light_direction()->set_z(sceneDescription.light.direction.z);

                dataset->append(sceneSetup);
            }

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
            dataset->batchAppend(gsl::make_span(samples));
            std::cout << "Finished writing samples." << std::endl;
        }
    }

    template <class T>
    void SceneSetupCollector::setupVariables(optix::Handle<T>& scope) const
    {
        scope["directionBuffer"]->setBuffer(directionBuffer);
        scope["positionBuffer"]->setBuffer(positionBuffer);
    }
}
