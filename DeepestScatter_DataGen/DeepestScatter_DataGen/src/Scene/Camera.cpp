#include "Camera.h"

#include "optix_math.h"

#pragma warning(push, 0)
#include "Util/sutil.h"
#pragma warning(pop)

#include "Util/Resources.h"
#include "GL/freeglut.h"
#include "Util/BufferBind.h"
#include "CUDA/rayData.cuh"
#include "SceneDescription.h"

namespace DeepestScatter
{
    static constexpr optix::uint2 RECT_SIZE{ 128, 128 };

    void Camera::init()
    {
        const std::string modelPath = "../../DeepestScatter_Train/runs/1024_mipmap_correct_logeps/checkpoint.pt";
        module = torch::jit::load(modelPath);

        const std::string programFile = "disneyCamera.cu";

        camera = resources->loadProgram(programFile, "pinholeCamera");
        context->setRayGenerationProgram(0, camera);

        updateFrameResult = resources->loadProgram(programFile, "updateFrameResult");
        clearRect = resources->loadProgram(programFile, "clearRect");
        clearScreen = resources->loadProgram(programFile, "clearScreen");

        cameraEye = optix::make_float3(2, -0.4f, 0);
        cameraLookat = optix::make_float3(0, 0, 0);
        cameraUp = optix::make_float3(0, 1, 0);

        cameraRotate = optix::Matrix4x4::identity();
        updatePosition();

        exception = resources->loadProgram(programFile, "exception");
        context->setExceptionProgram(0, exception);
        exception["errorColor"]->setFloat(123123123.123123123f, 0, 0);

        miss = resources->loadProgram(programFile, "miss");
        context->setMissProgram(0, miss);

        networkInputBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        networkInputBuffer->setElementSize(sizeof(Gpu::DisneyNetworkInput));

        directRadianceBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, RECT_SIZE.x, RECT_SIZE.y);
        directRadianceBuffer->setElementSize(sizeof(IntersectionInfo));

        frameResultBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        progressiveBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        varianceBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        screenBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);

        reinhardFirstPass = resources->loadProgram("reinhard.cu", "firstPass");
        reinhardSecondPass = resources->loadProgram("reinhard.cu", "secondPass");
        reinhardLastPass = resources->loadProgram("reinhard.cu", "applyReinhard");

        reinhardSumLuminanceColumn = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width);
        reinhardAverageLuminance = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);

        setupVariables(camera);
        setupVariables(updateFrameResult);
        setupVariables(clearScreen);
        setupVariables(clearRect);
        setupVariables(exception);
        setupVariables(miss);
        setupVariables(reinhardFirstPass);
        setupVariables(reinhardSecondPass);
        setupVariables(reinhardLastPass);
        
        reset();
    }

    void Camera::update()
    {
        if (!isCompleted())
        {
            updatePosition();
            render();
        }
    }

    void Camera::reset()
    {
        auto tmp = context->getRayGenerationProgram(0);

        subframeId = 0;
        context->setRayGenerationProgram(0, clearScreen);
        context->launch(0, width, height);

        context->setRayGenerationProgram(0, tmp);
    }

    bool Camera::isCompleted()
    {
        return completed;
    }

    void Camera::rotate(optix::float2 from, optix::float2 to)
    {
        cameraRotate = arcball.rotate(from, to);
        updatePosition();
        reset();
    }

    void Camera::updatePosition()
    {
        const float hfov = 30.0f;
        const float aspectRatio = static_cast<float>(width) /
            static_cast<float>(height);

        optix::float3 u, v, w;
        sutil::calculateCameraVariables(
            cameraEye, cameraLookat, cameraUp, hfov, aspectRatio,
            u, v, w, false);

        const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
            normalize(u),
            normalize(v),
            normalize(w),
            cameraLookat);
        const optix::Matrix4x4 transform = frame * cameraRotate * frame.inverse();

        cameraEye = make_float3(transform * make_float4(cameraEye, 1.0f));

        sutil::calculateCameraVariables(
            cameraEye, cameraLookat, cameraUp, hfov, aspectRatio,
            u, v, w, false);

        cameraRotate = optix::Matrix4x4::identity();

        camera["eye"]->setFloat(cameraEye);
        camera["U"]->setFloat(u);
        camera["V"]->setFloat(v);
        camera["W"]->setFloat(w);
    }

    void Camera::setupVariables(optix::Program& program)
    {
        program["networkInputBuffer"]->setBuffer(networkInputBuffer);
        program["directRadianceBuffer"]->setBuffer(directRadianceBuffer);
        program["frameResultBuffer"]->setBuffer(frameResultBuffer);

        program["progressiveBuffer"]->setBuffer(progressiveBuffer);
        program["varianceBuffer"]->setBuffer(varianceBuffer);

        program["totalPixels"]->setUint(width * height);
        program["screenBuffer"]->setBuffer(screenBuffer);

        program["sumLuminanceColumns"]->setBuffer(reinhardSumLuminanceColumn);
        program["averageLuminance"]->setBuffer(reinhardAverageLuminance);
    }

    void Camera::render()
    {
        uint32_t previousSubframe;
        if (context["subframeId"]->getType() != RT_OBJECTTYPE_UNKNOWN)
        {
            context["subframeId"]->getUint(previousSubframe);
        }

        //todo: more subframes
        for (int i = 0; i < 1; i++)
        {
            subframeId++;
            context["subframeId"]->setUint(subframeId);
            std::cout << "rendering subframe " << subframeId << std::endl;

            for (uint x = 0; x < width; x+= RECT_SIZE.x)
            {
                for (uint y = 0; y < height; y += RECT_SIZE.y)
                {
                    renderRect(optix::make_uint2(x, y));
                }
            }

            context->setRayGenerationProgram(0, updateFrameResult);
            context->validate();
            context->launch(0, width, height);
        }

        context->setRayGenerationProgram(0, reinhardFirstPass);
        context->launch(0, width, 1);

        context->setRayGenerationProgram(0, reinhardSecondPass);
        context->launch(0, 1, 1);

        context["exposure"]->setFloat(exposure);
        context->setRayGenerationProgram(0, reinhardLastPass);
        context->launch(0, width, height);

        GLenum glDataType = GL_UNSIGNED_BYTE;
        GLenum glFormat = GL_RGBA;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

        {
            BufferBind<optix::uchar4> screen(screenBuffer);
            glDrawPixels(width, height, glFormat, glDataType, static_cast<GLvoid*>(&screen[0]));
        }
    }

    void Camera::renderRect(optix::uint2 start)
    {
        context->setRayGenerationProgram(0, clearRect);
        context->launch(0, RECT_SIZE.x, RECT_SIZE.y);

        context["rectOrigin"]->setUint(start);

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

            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(torch::from_blob(&networkInput[0], {RECT_SIZE.x * RECT_SIZE.y, 10, 226 }).cuda());
            at::Tensor predicted = module->forward(inputs).toTensor().cpu();
            float* output = predicted.data<float>();

            uint subframe;
            context["subframeId"]->getUint(subframe);
            std::cout << "radiance " << output[0] << std::endl;
            uint id = 0;
            for (uint y = start.y; y < start.y + RECT_SIZE.y; y++)
            {
                for (uint x = start.x; x < start.x + RECT_SIZE.x; x++)
                {
                    if (intersectionInfo[id].hasScattered)
                    {
                        screen[y * width + x] = optix::make_float4(output[id]) + optix::make_float4(intersectionInfo[id].radiance);
                    }

                    id++;
                }
            }
        }
    }

    void Camera::increaseExposure()
    {
        exposure *= 1.2f;
    }

    void Camera::decreaseExposure()
    {
        exposure /= 1.2f;
    }
}
