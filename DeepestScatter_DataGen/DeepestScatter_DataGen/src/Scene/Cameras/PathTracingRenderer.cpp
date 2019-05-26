#include "PathTracingRenderer.h"

#include "Util/Resources.h"
#include "Util/BufferBind.h"
#include "CUDA/rayData.cuh"

namespace DeepestScatter
{
    optix::Program PathTracingRenderer::getCamera()
    {
        return camera;
    }

    void PathTracingRenderer::init()
    {
        const std::string cameraFile = "pathTracingCamera.cu";

        camera = resources->loadProgram(cameraFile, "pinholeCamera");
    }

    void PathTracingRenderer::render(optix::Buffer frameResultBuffer)
    {
        size_t width, height;
        frameResultBuffer->getSize(width, height);

        camera["frameResultBuffer"]->setBuffer(frameResultBuffer);

        context->setRayGenerationProgram(0, camera);
        context->validate();
        context->launch(0, width, height);
    }
}
