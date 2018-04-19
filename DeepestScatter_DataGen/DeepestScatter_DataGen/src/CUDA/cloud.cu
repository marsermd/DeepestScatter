#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#include "random.cuh"
#include "rayData.cuh"

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, minimalRayDistance, , );


// --------------- BOX SHAPE ------------------

static __device__ void makeBox(float3 & boxmin, float3  & boxmax) {
    float halfWidth = 0.5f;
    boxmin = make_float3(-halfWidth); boxmax = make_float3(halfWidth);
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

// --------------- Path Tracing ------------------

rtDeclareVariable(uint2, launchID, rtLaunchIndex, );

rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, resultRadiance, rtPayload, );

rtDeclareVariable(float1, sampleStep, , );
rtDeclareVariable(float1, opticalDensityMultiplier, , );
rtDeclareVariable(float3, lightDirection, , );
rtDeclareVariable(float , lightIntensity, , );
rtDeclareVariable(float3, lightColor, , );
rtDeclareVariable(float3, skyIntensity, , );
rtDeclareVariable(float3, groundIntensity, , );

rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> cloud;
rtTextureSampler<uchar1, 3, cudaReadModeNormalizedFloat> inScatter;
rtTextureSampler<uchar1, 1, cudaReadModeNormalizedFloat> mie;
rtTextureSampler<uchar1, 1, cudaReadModeNormalizedFloat> choppedMie;
rtTextureSampler<uchar1, 1, cudaReadModeNormalizedFloat> choppedMieIntegral;

rtDeclareVariable(unsigned int, subframeId, , );

static __host__ __device__ __inline__ bool isInBox(float3 pos)
{
    return pos.x >= 0.0f && pos.y >= 0.0f && pos.z >= 0.0f
        && pos.x <= 1.0f && pos.y <= 1.0f && pos.z <= 1.0f;
}

static __host__ __device__ __inline__ float getMiePhase(float cosTheta)
{
    return tex1D(mie, (cosTheta + 1) / 2).x;
}

static __host__ __device__ __inline__ float getChoppedMiePhase(float cosTheta)
{
    return tex1D(choppedMie, (cosTheta + 1) / 2).x;
}

rtDeclareVariable(float3, missColor, , );

RT_PROGRAM void closestHitRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += make_float3(0.5f);

    float3 radiance = make_float3(0);
    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    resultRadiance.result = make_float3(0);

    unsigned int seed = tea<4>(launchID.x * 800 + launchID.y, subframeId);
    float3 normalizedLightDirection = normalize(lightDirection);

    float russianRouletteCoefficient = 1;

    int depth = 0;
    while (isInBox(pos))
    {
        float3 stepAlongRay = direction * sampleStep.x;
        float transmitance = 1;
        float opticalDistance = rnd(seed);

        bool hasScattered = false;
        float3 scatterPos;
        float currentTransmit;
        while (isInBox(pos))
        {
            float extinction = tex3D(cloud, pos.x, pos.y, pos.z).x * opticalDensityMultiplier.x;

            currentTransmit = expf(-extinction);

            transmitance *= currentTransmit;

            pos += stepAlongRay;

            if (!hasScattered && opticalDistance > transmitance)
            {
                hasScattered = true;
                scatterPos = pos;
            }
        }

        float cosLightAngle = dot(-normalizedLightDirection, direction);
        float3 currentLight = make_float3(0);
        if (cosLightAngle > 0.9998918876f) // cos(9.35*1e-3 * pi / 2)
        {
            currentLight = lightColor * lightIntensity;
        }
        else
        {
            float t = clamp((direction.y + 0.5f) / 1.5f, 0.f, 1.f);
            currentLight = lerp(groundIntensity, skyIntensity, t);
        }
        radiance += currentLight * transmitance * russianRouletteCoefficient;

        if (hasScattered && isInBox(scatterPos))
        {
            pos = scatterPos;
            float rrThreshold = 0.8f;
            float rr = rnd(seed);
            if (rr > rrThreshold)
            {
                // bidirectional path tracing
                float3 inScatteredLight = lightColor * tex3D(inScatter, pos.x, pos.y, pos.z).x * getMiePhase(cosLightAngle) * 0.000272f;
                radiance += inScatteredLight * russianRouletteCoefficient;
                break;
            }
            else
            {
                russianRouletteCoefficient /= rrThreshold;

                float l = 0.f;
                float r = 1.f;
                float m = 0.5f;
                float val = rnd(seed);
                for (int i = 0; i < 16; i++)
                {
                    m = (l + r) / 2.f;
                    if (val > tex1D(choppedMieIntegral, m).x)
                    {
                        l = m;
                    }
                    else
                    {
                        r = m;
                    }
                }

                float cosTheta = (l + r) - 1;

                float3 newDirection = uniformOnSphereCircle(seed, cosTheta);

                Onb onb(direction);
                onb.inverse_transform(newDirection);

                direction = newDirection;
            }
        }
        depth++;
    }

    resultRadiance.result = radiance;

    {
        float3 stepAlongRay = ray.direction * sampleStep.x;
        float transmitance = 1;
        pos = hitPoint;
        while (isInBox(pos))
        {
            float extinction = tex3D(cloud, pos.x, pos.y, pos.z).x * opticalDensityMultiplier.x;
            float currentTransmit = expf(-extinction);
            transmitance *= currentTransmit;

            pos += stepAlongRay;
        }
        resultRadiance.importance = transmitance;
    }
}