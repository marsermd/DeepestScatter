#include <optix.h>

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include "../sutil/GL/glew.h"
#  if defined( _WIN32 )
#    include "../sutil/GL/wglew.h"
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <stdlib.h>
#include <string>
#include <iostream>
#include <memory>

#include "../sutil/sutil.h"

#include "Scene.h"

int width  = 512u;
int height = 512u;
std::shared_ptr<Scene> scene = nullptr;

// Mouse state
optix::int2    mousePrevPos;
int            mouseButton;

void printUsageAndExit(const char* argv0);

void glutDisplay();
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);

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

    glutMainLoop();
}

int main(int argc, char* argv[])
{
    try {

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
                // nothing special yet
            }
            else
            {
                fprintf(stderr, "Unknown option '%s'\n", argv[i]);
                printUsageAndExit(argv[0]);
            }
        }

        glutInitialize(&argc, argv);

        scene = std::shared_ptr<Scene>(new Scene(width, height, 1.f/512.f));
        scene->init();
        scene->addCloud(inputFile);

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
    scene->updateCamera();
    scene->display();
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

        scene->rotateCamera(fromScaled, toScaled);
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