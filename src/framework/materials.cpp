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
#include "framework/materials.h"
#include "config.h"
#include "framework/imaging.h"
#include "sampler/sampling.h"

fox_tracer::shading_data::shading_data(const vec3 &_x, const vec3 &n) noexcept
{
    x            = _x;
    g_normal     = n;
    s_normal     = n;
    surface_bsdf = nullptr;
}

bool fox_tracer::bsdf::base::is_light() const noexcept
{
    return emission.luminance() > 0.0f;
}

fox_tracer::color fox_tracer::bsdf::base::emit(const shading_data &sd, const vec3 &wi) const noexcept
{
    return emission;
}

void fox_tracer::bsdf::base::add_light(const color &_emission) noexcept
{
    emission = _emission;
}

fox_tracer::bsdf::diffuse::diffuse(texture *_albedo) noexcept
    : albedo(_albedo)
{}

fox_tracer::vec3 fox_tracer::bsdf::diffuse::sample(
    const shading_data &sd, sampler *s,
    color &reflected_colour, float &pdf)
{
    //~ Lambertian BRDF
    //~ rho_hd = integral H^2 fr * cos_theta d_omega = fr * pi

    //~ cosine-weighted, TODO: remove it for config switch later
    // vec3 wi_local = sampling::cosine_sample_hemisphere(s->next(), s->next());
    // pdf = sampling::cosine_hemisphere_pdf(wi_local);
    // reflected_colour = albedo->sample(sd.tu, sd.tv) / math::pi<float>;
    // return sd.shading_frame.to_world(wi_local);

    const auto mode = static_cast<hemisphere_sampling>(
                config().hemisphere_mode.load(std::memory_order_relaxed));

    vec3 wi_local;
    if (mode == hemisphere_sampling::uniform)
    {
        //~ p(wi) = 1 / (2*pi)
        //~ cos_theta = xi1, phi = 2*pi*xi2

        // TODO: Move to sampling namespace
        // const float xi1 = s->next();
        // const float xi2 = s->next();
        // const float cos_t = xi1;
        // const float sin_t = std::sqrt(std::max(0.0f, 1.0f - cos_t*cos_t));
        // const float phi   = 2.0f * math::pi<float> * xi2;
        // wi_local = vec3(sin_t*std::cos(phi), sin_t*std::sin(phi), cos_t);
        // pdf = 1.0f / (2.0f * math::pi<float>);

        wi_local = sampling::uniform_sample_hemisphere(s->next(), s->next());
        pdf      = sampling::uniform_hemisphere_pdf(wi_local);
    }
    else
    {
        //~ spherical-coord direct inverse CDF
        // cos_t = sqrt(1 - xi1), sin_t = sqrt(xi1)
        // wi_local = (sin_t*cos(phi), sin_t*sin(phi), cos_t)

        //~ Malley's method
        // TODO: try concentric disk warp for QMC
        wi_local = sampling::cosine_sample_hemisphere(s->next(), s->next());
        pdf      = sampling::cosine_hemisphere_pdf(wi_local);
    }

    //~ fr = rho / pi
    reflected_colour = albedo->sample(sd.tu, sd.tv) / math::pi<float>;
    return sd.shading_frame.to_world(wi_local);
}

fox_tracer::color fox_tracer::bsdf::diffuse::evaluate(
    const shading_data &sd, const vec3 &wi)
{
    return albedo->sample(sd.tu, sd.tv) / math::pi<float>;
}

float fox_tracer::bsdf::diffuse::pdf(
    const shading_data &sd, const vec3 &wi)
{
    const auto mode = static_cast<hemisphere_sampling>(
        config().hemisphere_mode.load(std::memory_order_relaxed));

    // const float cos_i = math::dot(wi, sd.s_normal);
    // if (cos_i <= 0.0f) return 0.0f;
    // return (mode == hemisphere_sampling::uniform)
    //      ? 1.0f / (2.0f * math::pi<float>)
    //      : cos_i / math::pi<float>;

    const vec3 wi_local = sd.shading_frame.to_local(wi);

    //~ p(wi) = 1 / (2*pi), cos_theta_i > 0
    if (mode == hemisphere_sampling::uniform)
        return sampling::uniform_hemisphere_pdf(wi_local);

    //~ p(wi) = cos_theta_i / pi, cos_theta_i > 0
    return sampling::cosine_hemisphere_pdf(wi_local);
}

float fox_tracer::bsdf::diffuse::mask(const shading_data &sd)
{
    //~ aliased edges TODO: try out something else
    // const float a = albedo->sample_alpha(sd.tu, sd.tv);
    // return a < 0.5f ? 0.0f : 1.0f;
    return albedo->sample_alpha(sd.tu, sd.tv);
}

bool fox_tracer::bsdf::diffuse::is_pure_specular() const
{
    return false;
}

bool fox_tracer::bsdf::diffuse::is_two_sided() const
{
    return true;
}

fox_tracer::bsdf::mirror::mirror(texture *_albedo) noexcept
    : albedo(_albedo)
{}

fox_tracer::vec3 fox_tracer::bsdf::mirror::sample(
    const shading_data &sd, sampler *s,
    color &reflected_colour, float &pdf)
{
    //~ Dirac delta BSDF refernce formula
    //~ fr(wo, wi) = rho * delta(wi - wr) / cos_theta_i
    //~ wr = reflect(wo, n) = (-wo.x, -wo.y, wo.z)

    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);

    //~ wr = 2(wo.n)n - wo
    // const vec3 n = sd.s_normal;
    // const vec3 wi_world = (n * (2.0f * math::dot(sd.wo, n))) - sd.wo;
    // return wi_world;

    const vec3 wi_local(-wo_local.x, -wo_local.y, wo_local.z);

    //~ TODO: handle wo_local.z <= 0 (back-face hit) - currently relies
    //~ on integrator's two-sided flip

    //~ delta cancel in MC ratio: fr * cos / pdf = rho
    //~ store as rho / cos so integrators cos theta multiply cancels
    const float cos_theta = std::max(math::epsilon<float>, std::fabs(wi_local.z));

    pdf = 1.0f;
    reflected_colour = albedo->sample(sd.tu, sd.tv) / cos_theta;

    return sd.shading_frame.to_world(wi_local);
}

fox_tracer::color fox_tracer::bsdf::mirror::evaluate(
    const shading_data &sd, const vec3 &wi)
{
    return {0.0f, 0.0f, 0.0f};
}

float fox_tracer::bsdf::mirror::pdf(const shading_data &sd, const vec3 &wi)
{
    return 0.0f;
}

float fox_tracer::bsdf::mirror::mask(const shading_data &sd)
{
    // return 1.0f;
    return albedo->sample_alpha(sd.tu, sd.tv);
}

bool fox_tracer::bsdf::mirror::is_pure_specular() const
{
    return true;
}

bool fox_tracer::bsdf::mirror::is_two_sided() const
{
    // test: just render black for now
    // return false;
    return true;
}

float fox_tracer::bsdf::fresnel::dielectric(
    float cos_theta, const float ior_int, const float ior_ext) noexcept
{
    float eta_i = ior_ext;
    float eta_t = ior_int;

    if (cos_theta < 0.0f)
    {
        const float tmp = eta_i; eta_i = eta_t; eta_t = tmp;
        cos_theta = -cos_theta;
    }

    if (cos_theta > 1.0f) cos_theta = 1.0f;

    const float sin_theta_i = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
    const float sin_theta_t = (eta_i / eta_t) * sin_theta_i;

    if (sin_theta_t >= 1.0f) return 1.0f;

    const float cos_theta_t = std::sqrt(std::max(0.0f, 1.0f - sin_theta_t * sin_theta_t));
    const float r_parl = ((eta_t * cos_theta) - (eta_i * cos_theta_t))
                       / ((eta_t * cos_theta) + (eta_i * cos_theta_t));
    const float r_perp = ((eta_i * cos_theta) - (eta_t * cos_theta_t))
                       / ((eta_i * cos_theta) + (eta_t * cos_theta_t));

    return 0.5f * (r_parl * r_parl + r_perp * r_perp);
}

fox_tracer::color fox_tracer::bsdf::fresnel::conductor(float cos_theta, const color &ior, const color &k) noexcept
{
    if (cos_theta < 0.0f) cos_theta = 0.0f;
    if (cos_theta > 1.0f) cos_theta = 1.0f;

    const float cos2 = cos_theta * cos_theta;
    const float sin2 = 1.0f - cos2;

    auto per_channel = [&](const float eta, const float kk) -> float
    {
        const float eta2 = eta * eta;
        const float k2   = kk * kk;
        const float t0   = eta2 - k2 - sin2;

        const float a2plusb2 = std::sqrt(std::max(0.0f,
                                  t0 * t0 + 4.0f * eta2 * k2));
        const float a2 = 0.5f * (a2plusb2 + t0);

        const float a  = std::sqrt(std::max(0.0f, a2));

        const float t1 = a2plusb2 + cos2;
        const float t2 = 2.0f * a * cos_theta;

        const float Rs = (t1 - t2) / (t1 + t2);
        const float t3 = cos2 * a2plusb2 + sin2 * sin2;
        const float t4 = t2 * sin2;
        const float Rp = Rs * (t3 - t4) / (t3 + t4);

        return 0.5f * (Rs + Rp);
    };

    return {
        per_channel(ior.red, k.red),
        per_channel(ior.green, k.green),
        per_channel(ior.blue, k.blue)
    };
}

float fox_tracer::bsdf::ggx::lambda(const vec3 &wi, const float alpha) noexcept
{
    const float cos2 = wi.z * wi.z;
    if (cos2 <= 0.0f) return 0.0f;

    const float tan2 = (1.0f - cos2) / cos2;
    return 0.5f * (-1.0f + std::sqrt(1.0f + alpha * alpha * tan2));
}

float fox_tracer::bsdf::ggx::g(const vec3 &wi, const vec3 &wo, const float alpha) noexcept
{
    return 1.0f / (1.0f + lambda(wi, alpha) + lambda(wo, alpha));
}

float fox_tracer::bsdf::ggx::d(const vec3 &h, const float alpha) noexcept
{
    const float cos2 = h.z * h.z;
    if (cos2 <= 0.0f) return 0.0f;

    const float a2 = alpha * alpha;
    const float denom = cos2 * (a2 - 1.0f) + 1.0f;

    return a2 / (math::pi<float> * denom * denom);
}
