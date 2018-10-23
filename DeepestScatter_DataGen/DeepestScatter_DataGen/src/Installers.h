#pragma once

#include "Hypodermic/Hypodermic.h"

namespace DeepestScatter
{
    Hypodermic::ContainerBuilder installPathTracingApp();

    Hypodermic::ContainerBuilder installSetupCollectorApp();

    Hypodermic::ContainerBuilder installDataset(const std::string& databasePath);

    Hypodermic::ContainerBuilder installFramework(
        const std::string& cloudPath,
        uint32_t width, uint32_t height);
}
