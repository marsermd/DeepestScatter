#ifndef RAY_DATA_CUH
#define RAY_DATA_CUH

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

struct PerRayData_radiance
{
    optix::float3 result;
    float importance;
    int depth;
};

#endif //RAY_DATA_CUH