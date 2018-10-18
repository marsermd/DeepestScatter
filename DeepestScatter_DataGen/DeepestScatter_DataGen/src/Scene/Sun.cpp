#include "Sun.h"

namespace DeepestScatter
{
    Sun::Sun(Settings settings, optix::Context context)
        : context(context),
          direction(settings.direction),
          color(settings.color),
          intensity(settings.intensity)
    {
    }

    void Sun::init()
    {
        context["lightDirection"]->setFloat(direction);
        context["lightColor"]->setFloat(color);
        context["lightIntensity"]->setFloat(intensity);
    }
}
