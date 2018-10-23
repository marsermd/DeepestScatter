#include "Hypodermic/Hypodermic.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>

#include "Util/sutil.h"

#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Util/Dataset/Dataset.h"
#include "Installers.h"
#include "ExecutionLoop/GuiExecutionLoop.h"
#include <filesystem>

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
        std::string cloudPath = argv[1];
        std::string databasePath = "../../DeepestScatter_Train/dataset.lmdb";

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
            auto dataset = datasetBuilder.build();

            std::queue<GuiExecutionLoop::LazyTask> tasks;

            std::filesystem::path p("../Clouds/10_FREEBIE_CLOUDS");
            for (std::filesystem::directory_iterator fileIt(p); !fileIt._At_end(); ++fileIt)
            {
                std::cout << fileIt->path() << " -> " << fileIt->path().extension() << std::endl;
                if (fileIt->path().has_extension() && ".vdb" == fileIt->path().extension())
                {
                    auto cloudPath = fileIt->path().string();

                    GuiExecutionLoop::LazyTask task = [=]()
                    {
                        di::ContainerBuilder builder;

                        builder.addRegistrations(installFramework(cloudPath, width, height));
                        builder.addRegistrations(installSetupCollectorApp());

                        auto container = builder.buildNestedContainerFrom(*dataset.get());

                        auto camera = container->resolve<Camera>();
                        camera->completed = false;

                        return container;
                    };

                    tasks.push(task);
                }
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