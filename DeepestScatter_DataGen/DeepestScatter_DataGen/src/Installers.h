#pragma once

#include "Hypodermic/Hypodermic.h"

#pragma warning (push, 0)
#include "SceneSetup.pb.h"
#pragma warning (pop)

namespace DeepestScatter
{
    Hypodermic::ContainerBuilder installPathTracingApp();

    Hypodermic::ContainerBuilder installSampleCollectorApp();
    Hypodermic::ContainerBuilder installRadianceCollectorApp();

    Hypodermic::ContainerBuilder installDataset(const std::string& databasePath);

    Hypodermic::ContainerBuilder installSceneSetup(const Storage::SceneSetup& sceneSetup, const std::string& cloudsRoot);

    Hypodermic::ContainerBuilder installFramework(int32_t sceneId, uint32_t width, uint32_t height);
}
