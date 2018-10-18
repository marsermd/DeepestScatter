#pragma once
#include "ExecutionLoop.h"

#include <optixu/optixu_math_namespace.h>
#include <memory>
#include <queue>

#include "Scene/Scene.h"
#include "Scene/Camera.h"

namespace DeepestScatter
{
    class GuiExecutionLoop : public ExecutionLoop
    {
    public:
        using Task = std::shared_ptr<boost::di::injector<std::shared_ptr<Scene>, std::shared_ptr<Camera>>>;

        GuiExecutionLoop(int argc, char** argv);
        ~GuiExecutionLoop();

        void getNextTask();
        void run(const std::queue<Task>& tasks);

    private:
        static void glutInitialize(int* argc, char** argv);
        static void glutDisplay();
        static void glutMousePress(int button, int state, int x, int y);
        static void glutMouseMotion(int x, int y);
        static void glutKeyboard(unsigned char key, int x, int y);
        static void glutRun();

        static GuiExecutionLoop* instance;

        uint32_t width = 640u;
        uint32_t height = 480u;

        // Mouse state
        optix::int2    mousePrevPos = optix::make_int2(0, 0);
        int            mouseButton = 0;

        std::shared_ptr<Scene> currentScene;
        std::shared_ptr<Camera> currentCamera;

        std::queue<Task> tasks;
    };

    inline void GuiExecutionLoop::run(const std::queue<Task>& tasks)
    {
        this->tasks = tasks;
        getNextTask();
        glutRun();
    }
}
