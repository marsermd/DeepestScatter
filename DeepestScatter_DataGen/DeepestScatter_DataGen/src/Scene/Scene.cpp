#include "Scene.h"

#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <iostream>
#include <fstream>

#include <gsl/gsl_util>

#include <optix.h>
#include "GL/freeglut.h"

#pragma warning(push, 0)
#include "Util/sutil.h"

#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>

#include <openvdb/Types.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/math/Stats.h>
#pragma warning(pop)

#include "SceneItem.h"
#include "Mie.h"

namespace DeepestScatter
{
    Scene::Scene(SampleStep sampleStep, const std::vector<std::shared_ptr<SceneItem>>& sceneItems, optix::Context context):
        sampleStep(sampleStep), 
        sceneItems(sceneItems),
        context(context)
    {
        auto lightDirection = optix::normalize(optix::make_float3(-0.586f, -0.766f, -0.2717f));
        context["lightDirection"]->setFloat(lightDirection);
        context["lightColor"]->setFloat(1.3f, 1.25f, 1.15f);
        context["lightIntensity"]->setFloat(6e5f);

        context["skyIntensity"]->setFloat(.6f, .6f, 2);
        context["groundIntensity"]->setFloat(.6f, .8f, 1);
        //context["skyIntensity"]->setFloat(0, 0, 0);
        //context["groundIntensity"]->setFloat(0, 0, 0);

        context->setRayTypeCount(2);
        context->setEntryPointCount(1);
    }

    void Scene::RestartProgressive()
    {
        std::cout << "restarting progressive" << std::endl;
        for each (auto item in sceneItems)
        {
            item->Reset();
        }
    }

    void Scene::Init()
    {
        context["sampleStep"]->setFloat(sampleStep);
        context["densityMultiplier"]->setFloat(cloudLengthMeters / meanFreePathMeters);

        context["mie"]->setTextureSampler(Mie::getMieSampler(context));
        context["choppedMie"]->setTextureSampler(Mie::getChoppedMieSampler(context));
        context["choppedMieIntegral"]->setTextureSampler(Mie::getChoppedMieIntegralSampler(context));

        for each (auto item in sceneItems)
        {
            item->Init();
        }
    }

    void Scene::Display()
    {
        for each (auto item in sceneItems)
        {
            item->Update();
        }
    }
}