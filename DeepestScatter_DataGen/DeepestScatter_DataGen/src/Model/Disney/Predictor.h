#pragma once

#include <filesystem>
#include "Util/Dataset/Dataset.h"

namespace DeepestScatter
{
    class DisneyPredictor
    {
    public:
        DisneyPredictor(const std::filesystem::path& model, std::shared_ptr<Dataset>& dataset);
    };
}
