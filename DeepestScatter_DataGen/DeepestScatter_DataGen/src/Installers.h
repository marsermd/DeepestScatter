#pragma once

#include "Hypodermic/Hypodermic.h"

#pragma warning (push, 0)
#include "SceneSetup.pb.h"
#pragma warning (pop)

#include "Scene/CloudMaterial.h"

namespace DeepestScatter
{
    Hypodermic::ContainerBuilder installApp();

    Hypodermic::ContainerBuilder installDataset(const std::string& databasePath);

    Hypodermic::ContainerBuilder installSceneSetup(
        const Persistance::SceneSetup& sceneSetup,
        const std::string& cloudsRoot,
        Cloud::Rendering::Mode renderingMode,
        Cloud::Model::Mipmaps mipmaps);

    Hypodermic::ContainerBuilder installFramework(uint32_t width, uint32_t height);
}
