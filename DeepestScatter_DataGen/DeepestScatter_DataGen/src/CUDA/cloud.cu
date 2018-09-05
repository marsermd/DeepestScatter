#include <optix_world.h>
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#include "random.cuh"
#include "rayData.cuh"
#include <ctime>

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, minimalRayDistance, , );

rtDeclareVariable(float3, boxSize, , );

// --------------- BOX SHAPE ------------------

static __device__ void makeBox(float3 & boxmin, float3  & boxmax) {
    boxmin = -boxSize / 2; boxmax = boxSize / 2;
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

rtDeclareVariable(float, sampleStep, , );
rtDeclareVariable(float, densityMultiplier, , );
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
    return pos.x >= -0.01f && pos.y >= -0.01f && pos.z >= -0.01f
        && pos.x <= boxSize.x + 0.01f && pos.y <= boxSize.y + 0.01f && pos.z <= boxSize.z + 0.01f;
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

static __host__ __device__ __inline__ float sampleCloud(float3 pos)
{
    pos = pos / boxSize;
    return tex3D(cloud, pos.x, pos.y, pos.z).x;
}

static __host__ __device__ __inline__ float sampleInScatter(float3 pos)
{
    pos = pos / boxSize;
    return tex3D(inScatter, pos.x, pos.y, pos.z).x;
}

RT_PROGRAM void closestHitRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * boxSize;

    float3 radiance = make_float3(0);
    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    resultRadiance.result = make_float3(0);

    unsigned int seed = tea<4>(launchID.x * 800 + launchID.y, subframeId);
    float3 normalizedLightDirection = normalize(lightDirection);

    int depth = 0;
    while (isInBox(pos))
    {
        depth++;

        float cosLightAngle = dot(-normalizedLightDirection, direction);

        float3 stepAlongRay = direction * sampleStep;
        float transmitance = 1;
        float opticalDistance = rnd(seed);

        bool hasScattered = false;
        float3 scatterPos;
        float currentTransmit;

        float skySampleProbability = 0.1f;
        bool sampleSky = subframeId % 10 == 0;

        while (isInBox(pos))
        {
            pos += stepAlongRay;

            float density = sampleCloud(pos) * densityMultiplier;
            float extinction = density * sampleStep;
            currentTransmit = expf(-extinction);
            transmitance *= currentTransmit;

            if (!hasScattered && opticalDistance > transmitance)
            {
                hasScattered = true;
                scatterPos = pos - direction * log(opticalDistance / transmitance) / density;
                
                if (!sampleSky)
                {
                    break;
                }
            }
        }

        if (sampleSky)
        {
            float3 currentLight;

            // We check that depth equals zero because for all other depths, 
            // the light from the sun is already taken into account by next event estimation 
            if (cosLightAngle > 0.99998930414f && depth == 1) // cos(0.53 / 180 * pi / 2)
            {
                currentLight = lightColor * lightIntensity;
            }
            else
            {
                float t = clamp((direction.y + 0.5f) / 1.5f, 0.f, 1.f);
                currentLight = lerp(groundIntensity, skyIntensity, t);
            }
            radiance += currentLight * transmitance / skySampleProbability;
        }

        if (hasScattered && isInBox(scatterPos))
        {
            {
                // next event estimation
                const float sunAngularRadiusDeg = 0.53f / 2;
                const float sphereArea = 4 * CUDART_PI_F;
                const float sunArea = 2 * CUDART_PI_F *(1 - cos(sunAngularRadiusDeg * CUDART_PI_F / 180.0f));
                const float sunToSphereAreaRatio = sunArea / sphereArea;

                float phase = depth == 1 ? getMiePhase(cosLightAngle) : getChoppedMiePhase(cosLightAngle);

                float3 inScatteredLight = lightColor * lightIntensity * sampleInScatter(scatterPos) * phase * sunToSphereAreaRatio;
                radiance += inScatteredLight;
            }
            pos = scatterPos;
            float rrThreshold = 1;
            float rr = rnd(seed);
            if (rr > rrThreshold || depth > 1000)
            {
                break;
            }
            else
            {
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

                direction = normalize(newDirection);
            }
        }
    }

    resultRadiance.result = radiance;
    resultRadiance.importance = 0;
}