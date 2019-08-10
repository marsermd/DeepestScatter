#include "BakedRenderer.h"

#include "Util/Resources.h"
#include "Util/BufferBind.h"
#include "CUDA/rayData.cuh"
#include "Util/optixExtraMath.h"
#include <torch/csrc/api/include/torch/utils.h>

namespace DeepestScatter
{
    static constexpr optix::uint2 RECT_SIZE{ 512, 256 };
    static const std::string modelDirectory = "../../DeepestScatter_Train/runs/Jun09_01-00-29_DESKTOP-D5QPR6V/";

    optix::Program BakedRenderer::getCamera()
    {
        return camera;
    }

    void BakedRenderer::init()
    {
        context->setPrintEnabled(true);

        const std::string renderModelPath = modelDirectory + "ProbeRendererModel.pt";
        renderModel = torch::jit::load(renderModelPath, torch::kCUDA);
        renderModel->eval();

        const std::string cameraFile = "bakedCamera.cu";
        camera = resources->loadProgram(cameraFile, "pinholeCamera");
        blit = resources->loadProgram(cameraFile, "copyToFrameResult");

        const auto options = torch::TensorOptions().device(torch::kCUDA, -1).requires_grad(false);
        auto lightProbeInput = torch::zeros({ RECT_SIZE.x * RECT_SIZE.y, 202 }, options);
        rendererInputs.emplace_back(lightProbeInput);

        lightProbeInputBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        lightProbeInputBuffer->setElementSize(sizeof(Gpu::LightProbeRendererInput));
        std::cout << std::endl << sizeof(Gpu::LightProbeRendererInput) << std::endl;
        lightProbeInputBuffer->setDevicePointer(context->getEnabledDevices()[0], lightProbeInput.data_ptr());

        auto descriptorInput = torch::zeros({ RECT_SIZE.x * RECT_SIZE.y, (int)Gpu::BakedRendererDescriptor::Descriptor::LAYERS_CNT, 226 }, options);
        rendererInputs.emplace_back(descriptorInput);
        descriptorInputBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        descriptorInputBuffer->setElementSize(sizeof(Gpu::BakedRendererDescriptor));
        std::cout << std::endl << sizeof(Gpu::BakedRendererDescriptor) << std::endl;

        descriptorInputBuffer->setDevicePointer(context->getEnabledDevices()[0], descriptorInput.data_ptr());

        const optix::float3 cloudSizeInMeanFreePath = context["bboxSize"]->getFloat3() * context["densityMultiplier"]->getFloat();
        lightProbeCount = fceil3_sz(cloudSizeInMeanFreePath / Gpu::LightProbe::STEP_IN_MEAN_FREE_PATH) + optix::make_size_t3(1);

        bakedLightProbes = Baker(context, resources, lightProbeCount).bake();
        context["bakedLightProbes"]->setBuffer(bakedLightProbes);

        directRadianceBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        directRadianceBuffer->setElementSize(sizeof(IntersectionInfo));

        predictedRadianceBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, RECT_SIZE.x, RECT_SIZE.y);

        setupVariables(camera);
        setupVariables(blit);
    }

    BakedRenderer::Baker::Baker(optix::Context context, std::shared_ptr<Resources> resources, optix::size_t3 probeCount):
        context(context), resources(resources), probeCount(probeCount)
    {
        const std::string lightProbeModelPath = modelDirectory + "LightProbeModel.pt";
        lightProbeModel = torch::jit::load(lightProbeModelPath);
        lightProbeModel->eval();

        const std::string bakeFile = "lightProbeBaker.cu";
        collect = resources->loadProgram(bakeFile, "collect");

        result = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, probeCount.x, probeCount.y, probeCount.z);
        result->setElementSize(sizeof(Gpu::LightProbe));

        const auto options = torch::TensorOptions().device(torch::kCUDA, -1).requires_grad(false);
        auto tensor = torch::zeros({ static_cast<int>(probeCount.x * probeCount.y), 10, 225 }, options);
        lightProbeInputs.emplace_back(tensor);
        descriptors = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, probeCount.x, probeCount.y);
        descriptors->setElementSize(sizeof(Gpu::LightMapNetworkInput));
        descriptors->setDevicePointer(context->getEnabledDevices()[0], tensor.data_ptr());

        collect["descriptors"]->setBuffer(descriptors);
    }

    optix::Buffer BakedRenderer::Baker::bake()
    {
        std::cout << "Baking Light Probes... " << probeCount.x << " " << probeCount.y << " " << probeCount.z << std::endl;

        torch::NoGradGuard noGradGuard;
        for (uint32_t i = 0; i < probeCount.z; i++)
        {
            bakeAtZ(i);
        }
        return result;
    }

    void BakedRenderer::Baker::bakeAtZ(const uint32_t posZ)
    {
        std::cout << "baking at z " << posZ << std::endl;
        collect["posZ"]->setUint(posZ);

        context->setRayGenerationProgram(0, collect);
        context->validate();
        context->launch(0, probeCount.x, probeCount.y);

        const size_t rectArea = probeCount.x * probeCount.y;
        const size_t rectSize = rectArea * sizeof(Gpu::LightProbe);

        {
            BufferBind<Gpu::LightProbe> lightProbes(result);

            at::Tensor predicted = lightProbeModel->forward(lightProbeInputs).toTensor();
            predicted = predicted.mul_(256).to(torch::kUInt8).cpu();
            uint8_t* output = predicted.data<uint8_t>();

            size_t width, height, depth;
            result->getSize(width, height, depth);
            
            std::memcpy(&lightProbes[0] + rectArea * posZ, output, rectSize);
        }
    }

    void BakedRenderer::setupVariables(optix::Program& program)
    {
        program["lightProbeInputBuffer"]->setBuffer(lightProbeInputBuffer);
        program["descriptorInputBuffer"]->setBuffer(descriptorInputBuffer);
        program["directRadianceBuffer"]->setBuffer(directRadianceBuffer);
        program["predictedRadianceBuffer"]->setBuffer(predictedRadianceBuffer);
    }

    void BakedRenderer::render(optix::Buffer frameResultBuffer)
    {
        torch::NoGradGuard noGradGuard;
        size_t width, height;
        frameResultBuffer->getSize(width, height);

        blit["frameResultBuffer"]->setBuffer(frameResultBuffer);
        camera["frameResultBuffer"]->setBuffer(frameResultBuffer);

        context->setEntryPointCount(2u);
        context->setRayGenerationProgram(0, camera);
        context->setRayGenerationProgram(1, blit);
        context->validate();

        for (size_t x = 0; x < width; x += RECT_SIZE.x)
        {
            for (size_t y = 0; y < height; y += RECT_SIZE.y)
            {
                renderRect(optix::make_uint2(x, y));
            }
        }
    }

    void BakedRenderer::renderRect(optix::uint2 start)
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

        at::Tensor predicted = renderModel->forward(rendererInputs).toTensor();
        predictedRadianceBuffer->setDevicePointer(context->getEnabledDevices()[0], predicted.data_ptr());
        
        context->launch(1, RECT_SIZE.x, RECT_SIZE.y);
    }
}
