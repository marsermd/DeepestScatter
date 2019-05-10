#include "Scene.h"

#include <algorithm>
#include <iostream>
#include <utility>

#pragma warning(push, 0)
#include "Util/sutil.h"
#pragma warning(pop)

#include "SceneItem.h"
#include "Mie.h"

namespace DeepestScatter
{
    Scene::Scene(std::vector<std::shared_ptr<SceneItem>> sceneItems, std::shared_ptr<optix::Context> context):
        sceneItems(std::move(sceneItems)),
        context(*context.get())
    {
        this->context["skyIntensity"]->setFloat(.1f, .2f, 2);
        this->context["groundIntensity"]->setFloat(.9f, 1.1f, 1.1f);

        this->context->setRayTypeCount(4);
        this->context->setEntryPointCount(1);
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
