#pragma once 

#include <math.h>
#include <optixu/optixu_math_namespace.h>
#include <cmath>


inline __host__ __device__ optix::int3&& fceil3(const optix::float3& a)
{
    return optix::make_int3(
        std::ceil(a.x), 
        std::ceil(a.y), 
        std::ceil(a.z)
    );
}

inline __host__ __device__ optix::size_t3&& fceil3_sz(const optix::float3& a)
{
    return optix::make_size_t3(
        std::ceil(a.x),
        std::ceil(a.y),
        std::ceil(a.z)
    );
}

inline __host__ __device__ optix::size_t3&& operator+(const optix::size_t3& a, const optix::size_t3& b)
{
    return optix::make_size_t3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}
