#include "DisneyRenderer.h"

#include "Util/Resources.h"
#include "Util/BufferBind.h"
#include "CUDA/rayData.cuh"

namespace DeepestScatter
{
    static constexpr optix::uint2 RECT_SIZE{ 128, 128 };

    optix::Program DisneyRenderer::getCamera()
    {
        return camera;
    }

    void DisneyRenderer::init()
    {
        const std::string modelPath = "../../DeepestScatter_Train/runs/DisneyModel/1024_mipmap_correct_logeps_1e3/checkpoint.pt";
        module = torch::jit::load(modelPath);
        module->eval();

        const std::string cameraFile = "disneyCamera.cu";

        camera = resources->loadProgram(cameraFile, "pinholeCamera");
        clearRect = resources->loadProgram(cameraFile, "clearRect");

        const auto options = torch::TensorOptions().device(torch::kCUDA, -1).requires_grad(false);
        auto tensor = torch::zeros({ RECT_SIZE.x * RECT_SIZE.y, 10, 226 }, options);
        networkInputs.emplace_back(tensor);

        networkInputBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        networkInputBuffer->setElementSize(sizeof(Gpu::DisneyNetworkInput));
        networkInputBuffer->setDevicePointer(context->getEnabledDevices()[0], tensor.data_ptr());

        directRadianceBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        directRadianceBuffer->setElementSize(sizeof(IntersectionInfo));

        setupVariables(camera);
        setupVariables(clearRect);
    }

    void DisneyRenderer::setupVariables(optix::Program& program)
    {
        program["networkInputBuffer"]->setBuffer(networkInputBuffer);
        program["directRadianceBuffer"]->setBuffer(directRadianceBuffer);
    }

    void DisneyRenderer::render(optix::Buffer frameResultBuffer)
    {
        size_t width, height;
        frameResultBuffer->getSize(width, height);

        for (size_t x = 0; x < width; x += RECT_SIZE.x)
        {
            for (size_t y = 0; y < height; y += RECT_SIZE.y)
            {
                renderRect(optix::make_uint2(x, y), frameResultBuffer);
            }
        }
    }

    void DisneyRenderer::renderRect(optix::uint2 start, optix::Buffer frameResultBuffer)
    {
        context["rectOrigin"]->setUint(start);
        camera["frameResultBuffer"]->setBuffer(frameResultBuffer);

        context->setRayGenerationProgram(0, camera);
        context->validate();
        context->launch(0, RECT_SIZE.x, RECT_SIZE.y);

        {
            BufferBind<Gpu::DisneyNetworkInput> networkInput(networkInputBuffer);
            BufferBind<IntersectionInfo> intersectionInfo(directRadianceBuffer);
            BufferBind<optix::float4> screen(frameResultBuffer);

            std::cout << screen[0].x << std::endl;

            bool hasActiveDescriptors = std::any_of(
                intersectionInfo.getData().begin(), intersectionInfo.getData().end(), [&](const auto& x) { return x.hasScattered; });

            for (int i = 0; i < intersectionInfo.getData().size(); i++)
            {
                hasActiveDescriptors |= intersectionInfo[i].hasScattered;
            }

            if (!hasActiveDescriptors)
            {
                return;
            }

            at::Tensor predicted = module->forward(networkInputs).toTensor();
            predicted = predicted.cpu();
            float* output = predicted.data<float>();

            size_t width, height;
            frameResultBuffer->getSize(width, height);

            size_t rectPixelId = 0;
            for (size_t y = start.y; y < start.y + RECT_SIZE.y; y++)
            {
                for (size_t x = start.x; x < start.x + RECT_SIZE.x; x++)
                {
                    if (intersectionInfo[rectPixelId].hasScattered)
                    {
                        screen[y * width + x] = optix::make_float4(output[rectPixelId]) + optix::make_float4(intersectionInfo[rectPixelId].radiance);
                    }

                    rectPixelId++;
                }
            }
        }
    }
}
