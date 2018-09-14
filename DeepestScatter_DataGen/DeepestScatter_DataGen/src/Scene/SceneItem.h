#pragma once

namespace DeepestScatter
{
    class SceneItem
    {
    public:
        virtual ~SceneItem() noexcept = default;

        virtual void Init() = 0;
        virtual void Update() = 0;
    };
} 