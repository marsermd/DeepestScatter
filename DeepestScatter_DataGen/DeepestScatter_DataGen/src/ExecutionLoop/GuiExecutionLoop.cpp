#include "GuiExecutionLoop.h"

#include <GL/glew.h>
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

namespace DeepestScatter
{
    GuiExecutionLoop* GuiExecutionLoop::instance = nullptr;

    void GuiExecutionLoop::glutInitialize(int* argc, char** argv)
    {
        glutInit(argc, argv);
        
        glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
        glutInitWindowSize(instance->width, instance->height);
        glutInitWindowPosition(100, 100);

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

        glutCreateWindow("Deepest Scatter");
        glutHideWindow();
    }

    void GuiExecutionLoop::glutRun()
    {
        // Initialize GL state
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 1, 0, 1, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glViewport(0, 0, instance->width, instance->height);

        glutShowWindow();
        glutReshapeWindow(instance->width, instance->height);

        // register glut callbacks
        glutDisplayFunc(glutDisplay);
        glutIdleFunc(glutDisplay);
        glutMouseFunc(glutMousePress);
        glutMotionFunc(glutMouseMotion);
        glutKeyboardFunc(glutKeyboard);

        glutMainLoop();
    }

    GuiExecutionLoop::GuiExecutionLoop(int argc, char** argv)
    {
        assert(instance == nullptr, "Two GuiExecutionsLoops running at once!");

        instance = this;

        try
        {
            glutInitialize(&argc, argv);
            glewInit();
        }
        catch (sutil::APIError& e) {
            std::cerr << "API error " << e.code << " " << e.file << ":" << e.line << std::endl;
            exit(1);
        }
    }

    GuiExecutionLoop::~GuiExecutionLoop()
    {
        instance = nullptr;
    }

    void GuiExecutionLoop::getNextTask()
    {
        if (tasks.empty())
        {
            glutLeaveMainLoop();
            return;
        }

        Task& task = tasks.front();
        tasks.pop();
        currentScene = task->create<std::shared_ptr<Scene>>();
        currentCamera = task->create<std::shared_ptr<Camera>>();

        currentScene->init();
    }

    void GuiExecutionLoop::glutDisplay()
    {
        if (instance->currentScene->isCompleted())
        {
            instance->getNextTask();
        }

        double t1 = sutil::currentTime();
        instance->currentScene->update();
        double t2 = sutil::currentTime();

        sutil::displayMillisecondsPerFrame((t2 - t1) * 1000);
        std::cout << "MS/FAME: " << (t2 - t1) * 1000 << std::endl;
        glutSwapBuffers();
    }

    void GuiExecutionLoop::glutMousePress(int button, int state, int x, int y)
    {
        if (state == GLUT_DOWN)
        {
            instance->mouseButton = button;
            instance->mousePrevPos = optix::make_int2(x, y);
        }
        else
        {
            // nothing
        }
    }

    void GuiExecutionLoop::glutKeyboard(unsigned char key, int x, int y)
    {
        switch (key)
        {
        case '=':
        case '+':
            instance->currentCamera->increaseExposure();
            break;
        case '-':
        case'_':
            instance->currentCamera->decreaseExposure();
            break;
        case' ':
            instance->currentCamera->completed = !instance->currentCamera->completed;
            break;
        default:
            break;
        }
    }

    void GuiExecutionLoop::glutMouseMotion(int x, int y)
    {
        if (instance->mouseButton == GLUT_LEFT_BUTTON)
        {
            const optix::float2 from =
            {
                static_cast<float>(instance->mousePrevPos.x),
                static_cast<float>(instance->mousePrevPos.y)
            };
            const optix::float2 to =
            {
                static_cast<float>(x),
                static_cast<float>(y)
            };

            const optix::float2 fromScaled = { from.x / instance->width, from.y / instance->height };
            const optix::float2 toScaled = { to.x / instance->width, to.y / instance->height };

            instance->currentCamera->rotate(fromScaled, toScaled);
        }

        instance->mousePrevPos = optix::make_int2(x, y);
    }
}
