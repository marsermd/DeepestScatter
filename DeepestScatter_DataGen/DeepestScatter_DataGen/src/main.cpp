#include <optix.h>
#include "Boost/di.hpp"

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include <stdlib.h>
#include <string>
#include <iostream>
#include <memory>

#include "Util/sutil.h"

#include "Scene/Scene.h"
#include "Scene/SceneItem.h"
#include "Scene/Camera.h"
#include "Scene/VDBCloud.h"
#include "Scene/CloudPTRenderer.h"

namespace di = boost::di;

uint32_t width  = 640u;
uint32_t height = 480u;

// Mouse state
optix::int2    mousePrevPos;
int            mouseButton;

std::shared_ptr<DeepestScatter::Scene> scene;
std::shared_ptr<DeepestScatter::Camera> camera;

void printUsageAndExit(const char* argv0);

void glutDisplay();
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutKeyboard(unsigned char key, int x, int y);

void glutInitialize(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Deepest Scatter");
    glutHideWindow();
}

void glutRun()
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow(width, height);

    // register glut callbacks
    glutDisplayFunc(glutDisplay);
    glutIdleFunc(glutDisplay);
    glutMouseFunc(glutMousePress);
    glutMotionFunc(glutMouseMotion);
    glutKeyboardFunc(glutKeyboard);

    glutMainLoop();
}

int main(int argc, char* argv[])
{
    try 
    {
        if (argc < 2)
        {
            printUsageAndExit(argv[0]);
        }
        std::string inputFile = argv[1];

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

        glutInitialize(&argc, argv);
        glewInit();

        try {
            auto injector = di::make_injector(
                di::bind<optix::Context>.to(optix::Context::create()),
                di::bind<DeepestScatter::Scene::SampleStep>.to(DeepestScatter::Scene::SampleStep{1.0f/512.f}),
                di::bind<DeepestScatter::Camera::Settings>.to(DeepestScatter::Camera::Settings{width, height}),
                di::bind<DeepestScatter::CloudPTRenderer::RenderMode>.to(DeepestScatter::CloudPTRenderer::RenderMode::Full),
                di::bind<DeepestScatter::Scene>.to<DeepestScatter::Scene>(),
                di::bind<DeepestScatter::SceneItem* []>.to
                <
                    DeepestScatter::VDBCloud, 
                    DeepestScatter::CloudPTRenderer,
                    DeepestScatter::Camera
                >()
            );
            scene = injector.create<std::shared_ptr<DeepestScatter::Scene>>();
            camera = injector.create<std::shared_ptr<DeepestScatter::Camera>>();
            injector.create<std::shared_ptr<DeepestScatter::VDBCloud>>()->setCloudPath(inputFile);
            scene->Init();
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            throw;
        }

        glutRun();

        return(0);
    }
    catch (sutil::APIError& e) {
        std::cerr << "API error " << e.code << " " << e.file << ":" << e.line << std::endl;
        exit(1);
    }
}

void glutDisplay()
{
    double t1 = sutil::currentTime();
    try {
        scene->Display();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }
    double t2 = sutil::currentTime();

    sutil::displayMillisecondsPerFrame((t2 - t1) * 1000);

    std::cout << "MS/FAME: " << (t2 - t1) * 1000 << std::endl;

    glutSwapBuffers();
}

void glutMousePress(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouseButton = button;
        mousePrevPos = optix::make_int2(x, y);
    }
    else
    {
        // nothing
    }
}

void glutKeyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case '=':
    case '+':
        camera->increaseExposure();
        break;
    case '-':
    case'_':
        camera->decreaseExposure();
        break;
    default:
        break;
    }
}

void glutMouseMotion(int x, int y)
{
    if (mouseButton == GLUT_LEFT_BUTTON)
    {
        const optix::float2 from = 
        { 
            static_cast<float>(mousePrevPos.x),
            static_cast<float>(mousePrevPos.y) 
        };
        const optix::float2 to = 
        { 
            static_cast<float>(x),
            static_cast<float>(y) 
        };

        const optix::float2 fromScaled = { from.x / width, from.y / height };
        const optix::float2 toScaled = { to.x / width, to.y / height };

        camera->Rotate(fromScaled, toScaled);
    }

    mousePrevPos = optix::make_int2(x, y);
}

void printUsageAndExit(const char* argv0)
{
    fprintf(stderr, "Usage  : %s [options]\n", argv0);
    fprintf(stderr, "Options: <filename> --show      Specify file for image output\n");
    fprintf(stderr, "         --help | -h                 Print this usage message\n");
    exit(1);
}