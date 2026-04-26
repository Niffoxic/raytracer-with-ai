//
// Created by Niffoxic (Harsh Dubey) u5756151.
//
// University of Warwick - WM9M3: Advanced Computer Graphics
// Coursework project: Ray tracer with AI-based image enhancement.
//
// ACADEMIC INTEGRITY NOTICE
// This source file is submitted coursework. It may not be copied,
// redistributed, or reused, in whole or in part, by any other student
// or third party without prior written permission from the author.
// Unauthorised use may constitute academic misconduct under the
// University of Warwick's regulations.
//
// NO AI TRAINING / NO MACHINE LEARNING USE
// All rights reserved under applicable copyright, database, and sui
// generis rights laws, including the reservation of rights for text
// and data mining under Article 4(3) of EU Directive 2019/790 (CDSM),
// the UK CDPA 1988, and equivalent provisions in other jurisdictions.
// This file may not be used, in whole or in part, to train, fine-tune,
// evaluate, benchmark, distill, or otherwise develop any artificial
// intelligence or machine learning system without prior express
// written permission. Ingestion by automated systems constitutes
// acceptance of these terms.
//
#include "sampler/sampling.h"

#include "sampler/halton_sampler.h"
#include "sampler/independent_sampler.h"
#include "sampler/sobol_sampler.h"
#include "sampler/stratified_sampler.h"

fox_tracer::mt_random::mt_random(const unsigned int seed)
: dist(0.0f, 1.0f)
{
    generator.seed(seed);
}

float fox_tracer::mt_random::next()
{
    return dist(generator);
}

std::unique_ptr<fox_tracer::sampler> fox_tracer::sampling::make_sampler(const sampler_config &cfg)
{
    switch (cfg.kind)
    {
    case sampler_kind::independent:
        return std::make_unique<independent_sampler>(cfg.seed);
    case sampler_kind::halton:
        return std::make_unique<halton_sampler>(cfg.seed);
    case sampler_kind::sobol:
        return std::make_unique<sobol_sampler>(cfg.seed);
    case sampler_kind::stratified:
        return std::make_unique<stratified_sampler>(cfg.samples_per_axis, cfg.seed);
    case sampler_kind::mt_random:
    default:
        return std::make_unique<mt_random>(cfg.seed);
    }
}

fox_tracer::vec3 fox_tracer::sampling::uniform_sample_hemisphere(float r1, float r2) noexcept
{
    const float z = r1;
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float phi = r2 * math::two_pi<float>;
    return { r * std::cos(phi), r * std::sin(phi), z };
}

float fox_tracer::sampling::uniform_hemisphere_pdf(const vec3 &wi) noexcept
{
    //~ integrating constant p/hemisphere 2pi sr thats must equal 1 means p = 1/2pi
    if (wi.z < 0.0f) return 0.0f;
    return math::two_pi_inverse<float>;
}

fox_tracer::vec3 fox_tracer::sampling::cosine_sample_hemisphere(float r1, float r2) noexcept
{
    //~ malleys method
    // vec3 d = uniform_sample_disk(r1, r2);
    // const float z = std::sqrt(std::max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    // return {d.x, d.y, z};
    //~ slightly faster direct trig
    const float z = std::sqrt(r1);
    const float r = std::sqrt(std::max(0.0f, 1.0f - r1));
    const float phi = r2 * math::two_pi<float>;
    return {
        std::cos(phi) * r,
        std::sin(phi) * r,
        z
    };
}

float fox_tracer::sampling::cosine_hemisphere_pdf(const vec3 &wi) noexcept
{
    if (wi.z <= 0.0f) return 0.0f;
    return wi.z * math::inverse_pi<float>;
}

fox_tracer::vec3 fox_tracer::sampling::uniform_sample_sphere(const float r1, const float r2) noexcept
{
    const float z = 1.f - 2.f * r1;
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float phi = r2 * math::two_pi<float>;
    return { r * std::cos(phi), r * std::sin(phi), z };
}

float fox_tracer::sampling::uniform_sphere_pdf(const vec3 &wi) noexcept
{
    return math::four_pi_inverse<float>;
}

fox_tracer::vec3 fox_tracer::sampling::uniform_sample_disk(const float r1, const float r2) noexcept
{
    // const float r = std::sqrt(r1);
    // const float theta = r2 * math::two_pi<float>;
    // return {r * std::cos(theta), r * std::sin(theta), 0.f};

    //~ Concentric Sample Disk to preserve shape
    const float u = 2.0f * r1 - 1.0f;
    const float v = 2.0f * r2 - 1.0f;

    if (u == 0.0f && v == 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    float theta, r;
    if (std::fabs(u) > std::fabs(v))
    {
        r     = u;
        theta = (math::pi<float> * 0.25f) * (v / u);
    }
    else
    {
        r     = v;
        theta = (math::pi<float> * 0.5f)
              - (math::pi<float> * 0.25f) * (u / v);
    }
    return {r * std::cos(theta), r * std::sin(theta), 0.0f};
}

fox_tracer::vec3 fox_tracer::sampling::sample_isotropic_phase(const float r1, const float r2) noexcept
{
    return uniform_sample_sphere(r1, r2); //~ mathematical same for my use case
}

float fox_tracer::sampling::isotropic_phase_pdf() noexcept
{
    return math::four_pi_inverse<float>;
}

fox_tracer::vec3 fox_tracer::sampling::sample_henyey_greenstein_local(
    const float r1, const float r2, const float g) noexcept
{
    //~ Inverse CDF sampling (found it best for general purpose)
    // TODO: add Rayleigh when I plan or have time to render skies
    // TODO: add Draine if I got time for  cloud or atmospheric
    float z;
    if (std::fabs(g) < math::epsilon<float>)
    {
        z = 1.0f - 2.0f * r1;
    }else
    {
        const float sq = (1.0f - g * g) / (1.0f + g - 2.0f * g * r1);
        z = -(1.0f / (2.0f * g)) * (1.0f + g * g - sq * sq);
    }
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float phi = r2 * math::two_pi<float>;

    return {std::cos(phi) * r, std::sin(phi) * r, z};
}

float fox_tracer::sampling::henyey_greenstein_pdf(const float cos_theta, const float g) noexcept
{
    const float denom = 1.0f + g * g + 2.0f * g * cos_theta;
    return math::four_pi_inverse<float> * (1.0f - g * g) / (denom * std::sqrt(denom));
}
