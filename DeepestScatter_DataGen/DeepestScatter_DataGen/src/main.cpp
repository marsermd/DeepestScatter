#include "Boost/di.hpp"

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>

#include "Util/sutil.h"

#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Util/Dataset/Dataset.h"
#include "SceneSetup.pb.h"
#include "Installers.h"
#include "Boost/uml_dumper.hpp"
#include "ExecutionLoop/GuiExecutionLoop.h"

namespace di = boost::di;
using namespace DeepestScatter;

uint32_t width = 640u;
uint32_t height = 480u;

void printUsageAndExit(const char* argv0);


int main(int argc, char* argv[])
{
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
            std::queue<GuiExecutionLoop::Task> tasks;

            for (int i = 0; i < 10; i++)
            {
                auto task = std::make_shared<di::injector<std::shared_ptr<Scene>, std::shared_ptr<Camera>>>(di::make_injector<DIConfig>
                (
                    installFramework(cloudPath, databasePath, width, height),
                    installSetupCollectorApp()
                ));

                task->create<std::shared_ptr<Camera>>()->completed = false;

                tasks.push(task);
            }

            GuiExecutionLoop loop(argc, argv);
            loop.run(tasks);
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