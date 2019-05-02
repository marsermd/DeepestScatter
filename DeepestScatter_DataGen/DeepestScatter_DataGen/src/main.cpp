#include "Hypodermic/Hypodermic.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>

#include "Util/sutil.h"

#include "Scene/Camera.h"
#include "installers.h"
#include "ExecutionLoop/GuiExecutionLoop.h"
#include "Util/Dataset/Dataset.h"

#pragma warning (push, 0)
#include "SceneSetup.pb.h"
#include "Result.pb.h"
#include "ExecutionLoop/Tasks.h"
#include "DisneyDescriptor.pb.h"
#pragma warning (pop)

namespace di = Hypodermic;
using namespace DeepestScatter;

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
        const std::string databasePath = "../../Data/Dataset/Train.lmdb";
        const std::string cloudRoot = "../../Data/Clouds_Train";

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
            GuiExecutionLoop loop(argc, argv);

            loop.run(Tasks::collect<Persistance::DisneyDescriptor>(databasePath, cloudRoot, Tasks::CollectMode::Reset));
            //loop.run(std::move(Tasks::renderCloud(cloudPath, 1000)));
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