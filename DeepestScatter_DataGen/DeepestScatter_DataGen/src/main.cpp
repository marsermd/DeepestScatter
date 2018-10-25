#include "Hypodermic/Hypodermic.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>

#include "Util/sutil.h"

#include "Scene/Camera.h"
#include "Installers.h"
#include "ExecutionLoop/GuiExecutionLoop.h"
#include "Util/Dataset/Dataset.h"

#pragma warning (push, 0)
#include "SceneSetup.pb.h"
#pragma warning (pop)

namespace di = Hypodermic;
using namespace DeepestScatter;

uint32_t width = 640u;
uint32_t height = 480u;

void printUsageAndExit(const char* argv0);



int main(int argc, char* argv[])
{
    di::Behavior::configureRuntimeRegistration(false);

    try
    {
        if (argc < 2)
        {
            printUsageAndExit(argv[0]);
        }
        const std::string cloudPath = argv[1];
        const std::string databasePath = "../../DeepestScatter_Train/dataset.lmdb";

        for (int i = 2; i < argc; i++)
        {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            {
                printUsageAndExit(argv[0]);
            }
            else if (strcmp(argv[i], "--show") == 0)
            { 
                // Nothing special yet.
            }
            else
            {
                std::cerr << "Unknown option " << argv[i] << std::endl;
                printUsageAndExit(argv[0]);
            }
        }

        try 
        {
            di::ContainerBuilder datasetBuilder;
            datasetBuilder.addRegistrations(installDataset(databasePath));
            auto rootContainer = datasetBuilder.build();

            auto dataset = rootContainer->resolve<Dataset>();

            std::queue<GuiExecutionLoop::LazyTask> tasks;
            
            size_t sceneCount = dataset->getRecordsCount<Storage::SceneSetup>();

            for (int32_t i = 0; i < sceneCount; i++)
            {
                auto sceneSetup = dataset->getRecord<Storage::SceneSetup>(i);

                GuiExecutionLoop::LazyTask task = [=]()
                {
                    di::ContainerBuilder taskBuilder;

                    taskBuilder.addRegistrations(installFramework(sceneSetup.cloud_path(), i, width, height));
                    taskBuilder.addRegistrations(installPathTracingApp());

                    auto container = taskBuilder.buildNestedContainerFrom(*rootContainer.get());

                    auto camera = container->resolve<Camera>();
                    camera->completed = false;

                    return container;
                };

                tasks.push(task);
            }

            GuiExecutionLoop loop(argc, argv);
            loop.run(std::move(tasks));
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            throw;
        }

        return(0);
    }
    catch (sutil::APIError& e) {
        std::cerr << "API error " << e.code << " " << e.file << ":" << e.line << std::endl;
        exit(1);
    }
}

void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n" << std::endl;
    std::cerr << "Options: <filename> --show      Specify file for image output\n" << std::endl;
    std::cerr << "         --help | -h                 Print this usage message\n" << std::endl;
    exit(1);
}