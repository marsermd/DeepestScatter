#include "DisneyRenderer.h"

#include "Util/Resources.h"
#include "Util/BufferBind.h"
#include "CUDA/rayData.cuh"
#include <torch/csrc/api/include/torch/utils.h>

namespace DeepestScatter
{
    static constexpr optix::uint2 RECT_SIZE{ 128, 128 };

    optix::Program DisneyRenderer::getCamera()
    {
        return camera;
    }

    void DisneyRenderer::init()
    {
        const std::string modelPath = "../../DeepestScatter_Train/runs/Jun09_22-12-48_DESKTOP-D5QPR6V/DisneyModel.pt";
        module = torch::jit::load(modelPath);
        module->eval();

        const std::string cameraFile = "disneyCamera.cu";

        camera = resources->loadProgram(cameraFile, "pinholeCamera");
        blit = resources->loadProgram(cameraFile, "copyToFrameResult");
        clearRect = resources->loadProgram(cameraFile, "clearRect");

        const auto options = torch::TensorOptions().device(torch::kCUDA, -1).requires_grad(false);
        auto tensor = torch::zeros({ RECT_SIZE.x * RECT_SIZE.y, 10, 226 }, options);
        networkInputs.emplace_back(tensor);

        networkInputBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        networkInputBuffer->setElementSize(sizeof(Gpu::DisneyNetworkInput));
        networkInputBuffer->setDevicePointer(context->getEnabledDevices()[0], tensor.data_ptr());

        directRadianceBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        directRadianceBuffer->setElementSize(sizeof(IntersectionInfo));

        predictedRadianceBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, RECT_SIZE.x, RECT_SIZE.y);

        setupVariables(camera);
        setupVariables(blit);
        setupVariables(clearRect);
    }

    void DisneyRenderer::setupVariables(optix::Program& program)
    {
        program["networkInputBuffer"]->setBuffer(networkInputBuffer);
        program["directRadianceBuffer"]->setBuffer(directRadianceBuffer);
        program["predictedRadianceBuffer"]->setBuffer(predictedRadianceBuffer);
    }

    void DisneyRenderer::render(optix::Buffer frameResultBuffer)
    {
        size_t width, height;
        frameResultBuffer->getSize(width, height);

        blit["frameResultBuffer"]->setBuffer(frameResultBuffer);
        camera["frameResultBuffer"]->setBuffer(frameResultBuffer);

        context->setEntryPointCount(2u);
        context->setRayGenerationProgram(0, camera);
        context->setRayGenerationProgram(1, blit);
        context->validate();

        torch::NoGradGuard noGradGuard;

        for (uint32_t x = 0; x < width; x += RECT_SIZE.x)
        {
            for (uint32_t y = 0; y < height; y += RECT_SIZE.y)
            {
                renderRect(optix::make_uint2(x, y), frameResultBuffer);
            }
        }
    }

    void DisneyRenderer::renderRect(optix::uint2 start, optix::Buffer frameResultBuffer)
    {
        context["rectOrigin"]->setUint(start);

        context->launch(0, RECT_SIZE.x, RECT_SIZE.y);

        {
            BufferBind<IntersectionInfo> intersectionInfo(directRadianceBuffer);

            const bool hasActiveDescriptors = std::any_of(
                intersectionInfo.getData().begin(), intersectionInfo.getData().end(), [&](const auto& x) { return x.hasScattered; });

            if (!hasActiveDescriptors)
            {
                return;
            }
        }

        at::Tensor predicted = module->forward(networkInputs).toTensor();
        predictedRadianceBuffer->setDevicePointer(context->getEnabledDevices()[0], predicted.data_ptr());

        context->launch(1, RECT_SIZE.x, RECT_SIZE.y);
    }
}
