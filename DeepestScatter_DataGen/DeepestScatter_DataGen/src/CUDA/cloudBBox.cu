#include "cloud.cuh"

static __device__ void makeBox(float3 & boxmin, float3 & boxmax) {
    boxmin = -bboxSize / 2; boxmax = bboxSize / 2;
}

RT_PROGRAM void intersect(int primIdx)
{
    float3 boxmin, boxmax;
    makeBox(boxmin, boxmax);

    float3 t0 = (boxmin - ray.origin) / ray.direction;
    float3 t1 = (boxmax - ray.origin) / ray.direction;
    float3 tnear = fminf(t0, t1);
    float3 tfar = fmaxf(t0, t1);
    float tmin = fmaxf(tnear);
    float tmax = fminf(tfar);

    if (tmin <= tmax)
    {
        bool checkBack = true;
        if (rtPotentialIntersection(tmin))
        {
            if (rtReportIntersection(0))
            {
                checkBack = false;
            }
        }
        if (checkBack)
        {
            if (rtPotentialIntersection(minimalRayDistance))
            {
                rtReportIntersection(0);
            }
        }
    }
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    float3 boxmin, boxmax;
    makeBox(boxmin, boxmax);
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->set(boxmin, boxmax);
}