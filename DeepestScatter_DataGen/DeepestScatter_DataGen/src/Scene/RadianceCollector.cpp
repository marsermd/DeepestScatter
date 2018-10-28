#include "RadianceCollector.h"

#include <optixu/optixu_math_namespace.h>
#include <numeric>

#include "Util/BufferBind.h"

#pragma warning(push, 0)
#include "Result.pb.h"
#include "ScatterSample.pb.h"
#include "CUDA/PointRadianceTask.h"
#include "Util/sutil.h"
#pragma warning(pop)

namespace DeepestScatter
{
    void RadianceCollector::init()
    {
        resetProgram = resources->loadProgram("pointEmissionCamera.cu", "clear");
        renderProgram = resources->loadProgram("pointEmissionCamera.cu", "estimateEmission");

        tasksBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, MAX_THREAD_COUNT);
        tasksBuffer->setElementSize(sizeof(PointRadianceTask));

        std::vector<PointRadianceTask> tasks;
        for (int i = 0; i < settings.batchSize; i++)
        {
            auto sample = dataset->getRecord<Storage::ScatterSample>(settings.batchStartId + i);
            const auto position = optix::make_float3
            (
                sample.point().x(),
                sample.point().y(),
                sample.point().z()
            );

            const auto direction = optix::make_float3
            (
                sample.view_direction().x(),
                sample.view_direction().y(),
                sample.view_direction().z()
            );

            tasks.emplace_back(PointRadianceTask(i, position, direction));
        }

        scheduleTasks(tasks);

        setupVariables(resetProgram);
        setupVariables(renderProgram);

        reset();
    }

    void RadianceCollector::reset()
    {
        frameId = 0;
        context->setRayGenerationProgram(0, resetProgram);
        context->launch(0, settings.batchSize);
    }

    int32_t RadianceCollector::getConvergedCount() const
    {
        return convergedTasks.size();
    }

    int32_t RadianceCollector::getRemainingCount() const
    {
        return settings.batchSize - getConvergedCount();
    }

    void RadianceCollector::update()
    {
        if (allPixelsConverged)
        {
            return;
        }

        uint32_t previousSubframe = 0;
        if (context["subframeId"]->getType() != RT_OBJECTTYPE_UNKNOWN)
        {
            context["subframeId"]->getUint(previousSubframe);
        }
        
        context->setRayGenerationProgram(0, renderProgram);
        double totalTime = 0;
        for (int i = 0; i < 100; i++)
        {
            frameId++;
            context["subframeId"]->setUint(frameId);
            double t1 = sutil::currentTime();
            context->launch(0, threadsCount);
            double t2 = sutil::currentTime();
            totalTime += t2 - t1;
        }
        std::cout << "MS/Render: " << totalTime * 1000 << " " << settings.batchSize - getConvergedCount() << std::endl;
        context["subframeId"]->setUint(previousSubframe);

        std::vector<PointRadianceTask> todoTasks;
        {
            BufferBind<PointRadianceTask> tasksBind(tasksBuffer);
            const uint32_t remainigCount = getRemainingCount();
            for (uint32_t i = 0; i < remainigCount; i++)
            {
                PointRadianceTask& representative = tasksBind[i * taskRepeatCount];
                for (uint32_t j = 1; j < taskRepeatCount; j++)
                {
                    representative += tasksBind[i * taskRepeatCount + j];
                }

                bool isConverged =
                    representative.getRelativeConfidenceInterval() < 0.02f ||
                    representative.getAbsoluteConfidenceInterval() < 1e-2f;
                if (isConverged)
                {
                    convergedTasks.push_back(representative);
                }
                else
                {
                    todoTasks.push_back(representative);
                }
            }
        }

        allPixelsConverged = getConvergedCount() == settings.batchSize;
        std::cout << "converged: " << getConvergedCount() << " of " << settings.batchSize << std::endl;

        if (allPixelsConverged)
        {
            recordToDataset();
        }
        else
        {
            scheduleTasks(todoTasks);
        }
    }

    bool RadianceCollector::isCompleted()
    {
        return allPixelsConverged;
    }

    void RadianceCollector::recordToDataset()
    {
        std::sort(convergedTasks.begin(), convergedTasks.end(),
            [&](const PointRadianceTask& a, const PointRadianceTask& b)
            {
                return a.id < b.id;
            }
        );

        std::vector<Storage::Result> results(settings.batchSize);
        std::cout << "Serializing emissions..." << std::endl;

        for (int i = 0; i < settings.batchSize; i++)
        {
            results[i].set_light_intensity(convergedTasks[i].radiance);
            results[i].set_is_converged(true);
            std::cout << convergedTasks[i].radiance << std::endl;
        }

        std::cout << "Writing emissions..." << std::endl;
        dataset->batchAppend(gsl::make_span(results), settings.batchStartId);
        std::cout << "Finished writing emissions." << std::endl;
    }

    void RadianceCollector::setupVariables(optix::Program& handle)
    {
        handle["tasks"]->setBuffer(tasksBuffer);
    }

    void RadianceCollector::scheduleTasks(const gsl::span<PointRadianceTask>& tasks)
    {
        taskRepeatCount = MAX_THREAD_COUNT / tasks.size();
        assert(taskRepeatCount > 0);

        threadsCount = tasks.size() * taskRepeatCount;

        BufferBind<PointRadianceTask> tasksBind(tasksBuffer);
        for (uint32_t i = 0; i < tasks.size(); i++)
        {
            tasksBind[i * taskRepeatCount] = tasks[i];
            for (uint32_t j = 1; j < taskRepeatCount; j++)
            {
                tasksBind[i * taskRepeatCount + j] = PointRadianceTask(tasks[i].id, tasks[i].position, tasks[i].direction);
            }
        }
    }
}
