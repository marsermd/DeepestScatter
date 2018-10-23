#include "Sun.h"

namespace DeepestScatter
{
    Sun::Sun(std::shared_ptr<Settings> settings, std::shared_ptr<optix::Context> context)
        : context(*context.get()),
          direction(settings->direction),
          color(settings->color),
          intensity(settings->intensity)
    {
    }

    void Sun::init()
    {
        context["lightDirection"]->setFloat(direction);
        context["lightColor"]->setFloat(color);
        context["lightIntensity"]->setFloat(intensity);
    }
}
