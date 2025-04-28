#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void lambertianDiffuse(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng
)
{
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);

    float cosTerm = glm::dot(normal, pathSegment.ray.direction);
    glm::vec3 f = m.color / PI;     // brdf = albedo / pi
    float pdf = 1 / PI * cosTerm;   // cosine-weight hemisphere sampling
    pathSegment.color *= (cosTerm * f / pdf);
}

__host__ __device__ void perfectMirror(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m
)
{
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);

    // Fresnel equation
    auto F0 = m.color;
    auto F90 = glm::vec3(1.0f, 1.0f, 1.0f);
    float cosTheta_i = glm::dot(normal, pathSegment.ray.direction);
    glm::vec3 fresnel = F0 + (F90 - F0) * (float)pow(1 - cosTheta_i, 5);

    // fr * cosθ = fresnel / cosθ * cosθ = fresnel
    // pdf = 1
    pathSegment.color *= fresnel;
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // If the material indicates that the object was a light
    if (m.emittance > 0.0f) {
        pathSegment.color *= (m.color * m.emittance);
        pathSegment.remainingBounces = 0;
    }
    else if (m.hasReflective) {
        perfectMirror(pathSegment, intersect, normal, m);
    }
    else {
        lambertianDiffuse(pathSegment, intersect, normal, m, rng);
    }
}

