#pragma once

#include "Hypodermic/Hypodermic.h"

namespace DeepestScatter
{
    Hypodermic::ContainerBuilder installPathTracingApp();

    Hypodermic::ContainerBuilder installSetupCollectorApp();

    Hypodermic::ContainerBuilder installFramework(
        const std::string& cloudPath,
        const std::string& databasePath,
        uint32_t width, uint32_t height);
}
