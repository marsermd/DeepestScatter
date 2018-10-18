#include "Camera.h"

#include "optix_math.h"

#pragma warning(push, 0)
#include "Util/sutil.h"
#pragma warning(pop)

#include "Util/Resources.h"
#include "GL/freeglut.h"
#include "Util/BufferBind.h"

namespace DeepestScatter
{
    void Camera::init()
    {
        camera = resources->loadProgram("pinholeCamera.cu", "pinholeCamera");
        context->setRayGenerationProgram(0, camera);

        clearScreen = resources->loadProgram("pinholeCamera.cu", "clearScreen");

        cameraEye = optix::make_float3(2, -0.4f, 0);
        cameraLookat = optix::make_float3(0, 0, 0);
        cameraUp = optix::make_float3(0, 1, 0);

        cameraRotate = optix::Matrix4x4::identity();
        updatePosition();

        exception = resources->loadProgram("pinholeCamera.cu", "exception");
        context->setExceptionProgram(0, exception);
        context["errorColor"]->setFloat(1, 0.6f, 0.6f);

        miss = resources->loadProgram("pinholeCamera.cu", "miss");
        context->setMissProgram(0, miss);

        progressiveBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        varianceBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
        screenBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);

        reinhardFirstPass = resources->loadProgram("reinhard.cu", "firstPass");
        reinhardSecondPass = resources->loadProgram("reinhard.cu", "secondPass");
        reinhardLastPass = resources->loadProgram("reinhard.cu", "applyReinhard");

        context["progressiveBuffer"]->setBuffer(progressiveBuffer);
        context["varianceBuffer"]->setBuffer(varianceBuffer);

        context["totalPixels"]->setUint(width * height);
        context["screenBuffer"]->setBuffer(screenBuffer);

        context["sumLogColumns"]->setBuffer(context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, width));
        context["lAverage"]->setBuffer(context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, 1));

        reset();
    }

    void Camera::update()
    {
        updatePosition();
        render();
    }

    void Camera::reset()
    {
        auto tmp = context->getRayGenerationProgram(0);

        subframeId = 0;
        context->setRayGenerationProgram(0, clearScreen);
        context->launch(0, width, height); //TODO: width, height from settings

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

    void Camera::render()
    {

        for (int i = 0; i < 10; i++)
        {
            subframeId++;
            context["subframeId"]->setUint(subframeId);
            context->setRayGenerationProgram(0, camera);
            context->validate();
            context->launch(0, width, height);

            context->setRayGenerationProgram(0, reinhardFirstPass);
            context->launch(0, width, 1);

            context->setRayGenerationProgram(0, reinhardSecondPass);
            context->launch(0, 1, 1);

            context["exposure"]->setFloat(exposure);
            context->setRayGenerationProgram(0, reinhardLastPass);
            context->launch(0, width, height);
        }

        GLenum glDataType = GL_UNSIGNED_BYTE;
        GLenum glFormat = GL_RGBA;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

        {
            BufferBind<optix::uint4> screen(screenBuffer);
            glDrawPixels(width, height, glFormat, glDataType, static_cast<GLvoid*>(&screen[0]));
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
