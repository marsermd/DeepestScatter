#include "Scene.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <optix.h>
#include "../sutil/sutil.h"

Scene::Scene(uint32_t width, uint32_t height, float sampleStep) :
    width(width), height(height), sampleStep(sampleStep), opticalDensityMultiplier(sampleStep * 512)
{
    context = optix::Context::create();

    context->setRayTypeCount(1);
    context->setEntryPointCount(1);

    screenBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height);
}


Scene::~Scene()
{
}


void Scene::addCloud(const std::string &path)
{
    cloudBuffer = loadVolumetricData(path);

    cloudSampler = createSamplerForBuffer3D(cloudBuffer);
    context["cloud"]->setTextureSampler(cloudSampler);

    RTsize cloudX, cloudY, cloudZ;
    cloudBuffer->getSize(cloudX, cloudY, cloudZ);

    optix::Buffer inScatterOut = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, cloudX, cloudY, cloudZ);

    optix::Program inScatter = loadProgram("inScatter.cu", "inScatter");
    inScatter["resultBuffer"]->setBuffer(inScatterOut);
    inScatter["lightDirection"]->setFloat(0.5, -0.8, 0.2);
    inScatter["lightIntensity"]->setFloat(1);
    context->setRayGenerationProgram(0, inScatter);

    context->validate();
    context->launch(0, cloudX, cloudY, cloudZ);

    cloudSampler->destroy();
    cloudBuffer->destroy();

    inScatterIn = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, cloudX, cloudY, cloudZ);
    float_t* to = (float_t*)inScatterIn->map();
    float_t* from = (float_t*)inScatterOut->map();
    std::memcpy(to, from, sizeof(float_t) * cloudX * cloudY * cloudZ);
    inScatterIn->unmap();
    inScatterOut->unmap();

    inScatterOut->destroy();
    cloudBuffer = loadVolumetricData(path);
    cloudSampler = createSamplerForBuffer3D(cloudBuffer);
    context["cloud"]->setTextureSampler(cloudSampler);

    inScatterSampler = createSamplerForBuffer3D(inScatterIn);
    context["inScatter"]->setTextureSampler(inScatterSampler);


    cloudGeometry = context->createGeometry();
    cloudGeometry->setBoundingBoxProgram(loadProgram("cloud.cu", "bounds"));
    cloudGeometry->setIntersectionProgram(loadProgram("cloud.cu", "intersect"));
    cloudGeometry->setPrimitiveCount(1u);
    cloudGeometry["minimalRayDistance"]->setFloat(0.001f);

    cloudMaterial = context->createMaterial();
    cloudMaterial->setClosestHitProgram(0, loadProgram("cloud.cu", "closestHitRadiance"));

    cloudInstance = context->createGeometryInstance(cloudGeometry, &cloudMaterial, &cloudMaterial + 1);
    cloudGroup = context->createGeometryGroup();
    cloudGroup->addChild(cloudInstance);
    cloudGroup->setAcceleration(context->createAcceleration("MedianBvh", "Bvh"));

    context["objectRoot"]->set(cloudGroup);
}

void Scene::init()
{
    context["sampleStep"]->setFloat(sampleStep);
    context["opticalDensityMultiplier"]->setFloat(opticalDensityMultiplier);

    camera = loadProgram("pinholeCamera.cu", "pinholeCamera");
    context["resultBuffer"]->setBuffer(screenBuffer);

    camera["eye"]->setFloat(0, 0, -1.5);
    camera["U"]->setFloat(1, 0, 0);
    camera["V"]->setFloat(0, 1, 0);
    camera["W"]->setFloat(0, 0, 1);

    exception = loadProgram("pinholeCamera.cu", "exception");
    context->setExceptionProgram(0, exception);
    context["errorColor"]->setFloat(1, 0.6, 0.6);

    miss = loadProgram("pinholeCamera.cu", "miss");
    context->setMissProgram(0, miss);
    context["missColor"]->setFloat(0.3, 0.3, 0.3);

    cameraEye = optix::make_float3(0, 0, -1.5);
    cameraLookat = optix::make_float3(0, 0, 0);
    cameraUp = optix::make_float3(0, 1, 0);

    cameraRotate = optix::Matrix4x4::identity();
}

void Scene::rotateCamera(optix::float2 from, optix::float2 to)
{
    cameraRotate = arcball.rotate(from, to);
}

void Scene::updateCamera()
{
    const float vfov = 60.0f;
    const float aspectRatio = static_cast<float>(width) /
        static_cast<float>(height);

    optix::float3 u, v, w;
    sutil::calculateCameraVariables(
        cameraEye, cameraLookat, cameraUp, vfov, aspectRatio,
        u, v, w);

    const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
        normalize(u),
        normalize(v),
        normalize(w),
        cameraLookat);
    const optix::Matrix4x4 transform = frame * cameraRotate * frame.inverse();

    cameraEye = make_float3(transform * make_float4(cameraEye, 1.0f));

    sutil::calculateCameraVariables(
        cameraEye, cameraLookat, cameraUp, vfov, aspectRatio,
        u, v, w, true);

    cameraRotate = optix::Matrix4x4::identity();

    camera["eye"]->setFloat(cameraEye);
    camera["U"]->setFloat(u);
    camera["V"]->setFloat(v);
    camera["W"]->setFloat(w);
}

void Scene::display()
{
    context->setRayGenerationProgram(0, camera);
    context["radianceRayType"]->setUint(0u);
    context->validate();
    context->launch(0, width, height);

    sutil::displayBufferGL(screenBuffer);
}

optix::Buffer Scene::loadVolumetricData(const std::string &path, bool border)
{
    std::ifstream fin(path, std::ios::binary);
    uint32_t sizeX, sizeY, sizeZ;
    fin.read(reinterpret_cast<char*>(&sizeX), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&sizeY), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&sizeZ), sizeof(uint32_t));

    uint32_t totalSize = sizeX * sizeY * sizeZ;

    std::cout << "Creating buffer" << std::endl;

    optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT);
    buffer->setSize(sizeX, sizeY, sizeZ);

    {
        float_t* density = (float_t*)buffer->map();

        std::cout << "Trying to read volumetric data of " << sizeX << "x" << sizeY << "x" << sizeZ << " with total elements " << totalSize << std::endl;
        fin.read(reinterpret_cast<char*>(&density[0]), sizeof(float_t) * totalSize);
        std::cout << "Read has completed successfuly" << std::endl;

        int pos = 0;
        for (int i = 0; i < sizeZ; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                for (int k = 0; k < sizeX; k++)
                {
                    bool isFace = false;
                    isFace |= i == 0 || i == sizeZ - 1;
                    isFace |= j == 0 || j == sizeY - 1;
                    isFace |= k == 0 || k == sizeX - 1;

                    if (isFace)
                    {
                        density[pos] = 0;
                    }
                    else
                    {
                        /*
                        Converting physical density to optical density. This allows to make calculations independent from step size.
                        An optical density can be derived from a real density in the following way:
                        opticalDensity = -log(1 - density)

                        Let's prove it.

                        assert(transmittance == 1 - absorbed)

                        After one sample of a volume absorbance is changed as follows:
                        absorbed' = absorbed + density * (1 - absorbed) 
                                  = absorbed * (1 - density) + density

                        And transmittance:
                        transmittance' = transmittance * e^-opticalDensity

                        Now Let's derive equation for absorbance from transmittance.
                        1 - absorbed' = (1 - absorbed) * e^-opticalDensity 
                                      = e^-opticalDensity - absorbed * e^-opticalDensity

                        And therefore
                        absorbed' = absorbed * e^-opticalDensity - e^-opticalDensity + 1 
                                  = absorbed * (1 - density) + density

                        Q.E.D.
                        */
                        density[pos] = -logf(1 - density[pos] / 255.0f);
                    }


                    pos++;
                }
            }
        }

        buffer->unmap();
    }

    return buffer;
}

optix::TextureSampler Scene::createSamplerForBuffer3D(optix::Buffer buffer)
{
    optix::TextureSampler sampler3D = context->createTextureSampler();

    for (uint32_t dim = 0; dim < 3; dim++)
    {
        sampler3D->setWrapMode(dim, RT_WRAP_CLAMP_TO_EDGE);
    }

    sampler3D->setFilteringModes(
        RT_FILTER_LINEAR,
        RT_FILTER_LINEAR,
        RT_FILTER_NONE
    );

    sampler3D->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler3D->setBuffer(buffer);

    return sampler3D;
}

optix::Program Scene::loadProgram(const std::string &fileName, const std::string programName)
{
    const std::string ptxPath = "./CUDA/";
    const std::string ptxExtension = ".ptx";
    
    std::string path = ptxPath + fileName + ptxExtension;

    std::cout << "loading program " << path;

    return context->createProgramFromPTXFile(path, programName);
}