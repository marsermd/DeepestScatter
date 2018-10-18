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
    Scene::Scene(const std::vector<std::shared_ptr<SceneItem>>& sceneItems, optix::Context context):
        sceneItems(sceneItems),
        context(context)
    {
        context["skyIntensity"]->setFloat(.6f, .6f, 2);
        context["groundIntensity"]->setFloat(.6f, .8f, 1.1f);

        context->setRayTypeCount(2);
        context->setEntryPointCount(1);
    }

    void Scene::restartProgressive()
    {
        std::cout << "restarting progressive" << std::endl;
        for (const auto& item : sceneItems)
        {
            item->reset();
        }
    }

    void Scene::init()
    {
        context["mie"]->setTextureSampler(Mie::getMieSampler(context));
        context["choppedMie"]->setTextureSampler(Mie::getChoppedMieSampler(context));
        context["choppedMieIntegral"]->setTextureSampler(Mie::getChoppedMieIntegralSampler(context));

        for (const auto& item : sceneItems)
        {
            item->init();
        }
    }

    void Scene::update()
    {
        for (const auto& item: sceneItems)
        {
            item->update();
        }
    }

    bool Scene::isCompleted()
    {
        return std::all_of(sceneItems.begin(), sceneItems.end(), [](auto x)
        {
            return x->isCompleted();
        });
    }
}
