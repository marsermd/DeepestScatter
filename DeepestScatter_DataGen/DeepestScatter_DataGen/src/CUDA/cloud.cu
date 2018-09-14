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

struct ScatteringEvent
{
    bool hasScattered;
    float3 scatterPos;
    float transmittance;
};

static __device__ __inline__ ScatteringEvent getNextScatteringEvent(
    unsigned int& seed,
    float3 pos, const float3& direction, bool stopAtScatterPos = true)
{
    float3 stepAlongRay = direction * sampleStep;
    float opticalDistance = rnd(seed);

    float transmittance = 1;
    bool hasScattered = false;
    float3 scatterPos = make_float3(0);

    while (isInBox(pos))
    {
        pos += stepAlongRay;

        float density = sampleCloud(pos) * densityMultiplier;
        float extinction = density * sampleStep;
        float currentTransmit = expf(-extinction);
        transmittance *= currentTransmit;

        if (!hasScattered && opticalDistance > transmittance)
        {
            hasScattered = true;
            scatterPos = pos - direction * log(opticalDistance / transmittance) / density;

            if (stopAtScatterPos)
            {
                break;
            }
        }
    }

    if (!hasScattered && !isInBox(pos))
    {
        scatterPos = pos;
    }

    return { hasScattered, scatterPos, transmittance };
}

static __device__ __inline__ const float3& sampleSky(const ScatteringEvent& scatter, const float3& direction)
{
    float3 currentLight;

    float t = clamp((direction.y + 0.5f) / 1.5f, 0.f, 1.f);
    currentLight = lerp(groundIntensity, skyIntensity, t);

    return currentLight * scatter.transmittance;
}

static __device__ __inline__ const float3& sampleSun(const float3& direction)
{
    float cosLightAngle = dot(-lightDirection, direction);

    if (cosLightAngle > 0.99998930414f) // cos(0.53 / 180 * pi / 2)
    {
        return lightColor * lightIntensity;
    }

    return make_float3(0);
}

static __device__ __inline__ const float3& getInScattering(const ScatteringEvent& scatter, const float3& direction, bool choppedMiePhase)
{
    constexpr float sunAngularRadiusDeg = 0.53f / 2;
    constexpr float sphereArea = 4 * CUDART_PI_F;
    const float sunArea = 2 * CUDART_PI_F *(1 - cos(sunAngularRadiusDeg * CUDART_PI_F / 180.0f));
    const float sunToSphereAreaRatio = sunArea / sphereArea;

    float cosLightAngle = dot(-lightDirection, direction);

    float phase = choppedMiePhase ? getChoppedMiePhase(cosLightAngle) : getMiePhase(cosLightAngle);

    return lightColor * lightIntensity * sampleInScatter(scatter.scatterPos) * phase * sunToSphereAreaRatio;
}

static __device__ __inline__ const float3& getNewDirection(unsigned int& seed, const float3& previousDirection)
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

    Onb onb(previousDirection);
    onb.inverse_transform(newDirection);

    return normalize(newDirection);
}

RT_PROGRAM void closestHitRadiance()
{
    float3 hitPoint = ray.origin + tHit * ray.direction;
    hitPoint += 0.5f * boxSize;

    float3 radiance = make_float3(0);
    float3 pos = hitPoint;

    float3 direction = normalize(ray.direction);

    resultRadiance.result = make_float3(0);

    unsigned int seed = tea<4>(launchID.x * 4096 + launchID.y, subframeId);
    
    float skySampleProbability = 0.1f;
    bool shouldSampleSky = subframeId % 10 == 0;

    int depth = 0;
    while (isInBox(pos))
    {
        depth++;
        if (depth == 1000)
        {
            break;
        }

        ScatteringEvent scatter =  getNextScatteringEvent(seed, pos, direction, !shouldSampleSky);

        if (shouldSampleSky)
        {
            // We check that depth equals one because for all other depths, 
            // the light from the sun is already taken into account by next event estimation
            if (depth == 1)
            {
                radiance += sampleSun(direction) / skySampleProbability;
            }
            radiance += sampleSky(scatter, direction) / skySampleProbability;
        }

        if (!scatter.hasScattered || !isInBox(scatter.scatterPos))
        {
            break;
        }
        else
        {
            // next event estimation
            radiance += getInScattering(scatter, direction, depth != 1);

            pos = scatter.scatterPos;

            direction = getNewDirection(seed, direction);
        }
    }

    resultRadiance.result = radiance;
    resultRadiance.importance = 0;
}