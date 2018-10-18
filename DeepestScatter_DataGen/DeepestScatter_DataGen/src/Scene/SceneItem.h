#pragma once

namespace DeepestScatter
{
    class SceneItem
    {
    public:
        virtual ~SceneItem() noexcept = default;

        virtual void init() = 0;
        virtual void reset() = 0;
        virtual void update() = 0;

        virtual bool isCompleted()
        {
            return true;
        }
    };
} 