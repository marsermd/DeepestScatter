#include "EmptyRenderer.h"

namespace DeepestScatter
{
    optix::Program EmptyRenderer::getCamera()
    {
        return optix::Program();
    }
}
