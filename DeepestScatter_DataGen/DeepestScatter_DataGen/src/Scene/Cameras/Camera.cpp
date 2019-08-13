#include "Camera.h"

#pragma warning(push, 0)
#include "Util/sutil.h"
#pragma warning(pop)

#include "Util/Resources.h"
#include "GL/freeglut.h"
#include "Util/BufferBind.h"
#include "CUDA/rayData.cuh"

#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfArray.h>


namespace DeepestScatter
{
    void Camera::init()
    {
        renderer->init();

        const std::string progressiveFile = "progressive.cu";
        updateFrameResult = resources->loadProgram(progressiveFile, "updateFrameResult");
        clearScreen = resources->loadProgram(progressiveFile, "clearScreen");
        exception = resources->loadProgram(progressiveFile, "exception");
        miss = resources->loadProgram(progressiveFile, "miss");

        context->setExceptionProgram(0, exception);
        context->setMissProgram(0, miss);

        exception["errorColor"]->setFloat(123123123.123123123f, 0, 0);

        cameraEye = optix::make_float3(2.5f, -0.4f, 0);
        cameraLookat = optix::make_float3(0, 0, 0);
        cameraUp = optix::make_float3(0, 1, 0);

        cameraRotate = optix::Matrix4x4::identity();
        updatePosition();


        frameResultBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        progressiveBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        varianceBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        screenBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);

        reinhardFirstPass = resources->loadProgram("reinhard.cu", "firstPass");
        reinhardSecondPass = resources->loadProgram("reinhard.cu", "secondPass");
        reinhardLastPass = resources->loadProgram("reinhard.cu", "applyReinhard");

        reinhardSumLuminanceColumn = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width);
        reinhardAverageLuminance = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);

        setupVariables(updateFrameResult);
        setupVariables(clearScreen);
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

        optix::Program camera = renderer->getCamera();
        if (camera.get() != nullptr)
        {
            camera["eye"]->setFloat(cameraEye);
            camera["U"]->setFloat(u);
            camera["V"]->setFloat(v);
            camera["W"]->setFloat(w);
        }
    }

    void Camera::setupVariables(optix::Program& program)
    {        
        program["frameResultBuffer"]->setBuffer(frameResultBuffer);
        program["progressiveBuffer"]->setBuffer(progressiveBuffer);
        program["varianceBuffer"]->setBuffer(varianceBuffer);

        program["totalPixels"]->setUint(width * height);
        program["screenBuffer"]->setBuffer(screenBuffer);

        program["sumLuminanceColumns"]->setBuffer(reinhardSumLuminanceColumn);
        program["averageLuminance"]->setBuffer(reinhardAverageLuminance);
    }

    void Camera::saveToDisk() const
    {
        using namespace Imath;
        using namespace Imf;

        Header header(width, height, 1, V2f(0, 0), 1, DECREASING_Y);
        header.channels().insert("R", Channel(Imf::FLOAT));
        header.channels().insert("G", Channel(Imf::FLOAT));
        header.channels().insert("B", Channel(Imf::FLOAT));

        BufferBind<optix::float4> frame(progressiveBuffer);

        float* start = reinterpret_cast<float*>(&frame[0]);
        const size_t xStride = sizeof(optix::float4);
        const size_t yStride = sizeof(optix::float4) * width;

        std::cout << frame[width * height / 2 + width / 2].x << std::endl;

        OutputFile file(outputFile.string().c_str(), header);
        FrameBuffer frameBuffer;
        frameBuffer.insert("R", Slice(Imf::FLOAT, reinterpret_cast<char*>(start++), xStride, yStride));
        frameBuffer.insert("G", Slice(Imf::FLOAT, reinterpret_cast<char*>(start++), xStride, yStride));
        frameBuffer.insert("B", Slice(Imf::FLOAT, reinterpret_cast<char*>(start++), xStride, yStride));
        file.setFrameBuffer(frameBuffer);
        
        file.writePixels(gsl::narrow<int>(height));
    }

    void Camera::render()
    {
        if (!isConverged())
        {
            //TODO: previous subframe???
            uint32_t previousSubframe;
            if (context["subframeId"]->getType() != RT_OBJECTTYPE_UNKNOWN)
            {
                context["subframeId"]->getUint(previousSubframe);
            }

            //todo: configurable subframe count
            for (int i = 0; i < 10; i++)
            {
                subframeId++;
                context["subframeId"]->setUint(subframeId);
                std::cout << "rendering subframe " << subframeId << std::endl;

                renderer->render(frameResultBuffer);

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
            if (subframeId % 40 == 0)
            {
                saveToDisk();
            }
        }
        else
        {
            completed = true;
            std::cout << "rendering subframe " << subframeId << std::endl;
        }

        GLenum glDataType = GL_UNSIGNED_BYTE;
        GLenum glFormat = GL_RGBA;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

        {
            BufferBind<optix::uchar4> screen(screenBuffer);
            glDrawPixels(width, height, glFormat, glDataType, static_cast<GLvoid*>(&screen[0]));
        }
    }

    bool Camera::isConverged()
    {
        if (subframeId < 100)
        {
            return false;
        }

        BufferBind<optix::float4> runningVariance(varianceBuffer);
        BufferBind<optix::float4> progressive(progressiveBuffer);

        bool isConverged = true;

        size_t convergedCount = 0;
        for (size_t id = 0; id < runningVariance.getData().size(); id++)
        {
            const float pixelRunningVariance = runningVariance[id].x;
            const float N = gsl::narrow<float>(subframeId);
            const float sigma = sqrtf(pixelRunningVariance / N);
            const float absoluteConfidence = 1.96f * sigma / sqrtf(N);
            const float relativeConfidence = absoluteConfidence / (progressive[id].x + FLT_EPSILON);

            const bool isConvergedPixel =
                relativeConfidence < 0.02f ||
                absoluteConfidence < 1e-2f;
            isConverged &= isConvergedPixel;

            if (isConvergedPixel)
            {
                convergedCount++;
            }
        }
        std::cout << "Converged: " << convergedCount << "/" << runningVariance.getData().size() 
            << " --- " << runningVariance.getData().size() - convergedCount << "left" << std::endl;

        //TODO:
        return runningVariance.getData().size() - convergedCount < 500;
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
