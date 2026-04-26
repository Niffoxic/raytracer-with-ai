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

// TODO: Add Test Flag so that I can check spec and two sided with false or true without manually changing them!!!

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

fox_tracer::bsdf::conductor::conductor(
    texture *_albedo, const color &_eta,
    const color &_k, const float roughness) noexcept
    : albedo(_albedo), eta(_eta), k(_k), alpha(1.62142f * std::sqrt(roughness))
{
    //~ TODO: try alpha = r*r the Burley one
}

fox_tracer::vec3 fox_tracer::bsdf::conductor::sample(
    const shading_data &sd, sampler *s, color &reflected_colour,
    float &pdf)
{
    //~ GGX microfacet conductor reference
    //~ fr = F * D * G / (4 * cos theta o * cos theta i)
    //~ F = fc(wo.h: should be abs if I remember correctly, eta, k)
    //~ D = a^2 / (pi * (cos_theta_h^2 * (a^2 - 1) + 1)^2)
    //~ G = 1 / (1 + lambda(wi) + lambda(wo))

    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);

    if (wo_local.z <= 0.0f)
    {
        pdf = 0.0f;
        reflected_colour = color(0.0f, 0.0f, 0.0f);
        return sd.wo;
    }

    const color tint = albedo->sample(sd.tu, sd.tv);

    if (alpha < math::epsilon<float>)
    {
        const vec3 wi_local(-wo_local.x, -wo_local.y, wo_local.z);

        const float cos_theta = std::max(
            math::epsilon<float>,
            std::fabs(wi_local.z)
        );
        //~ at normal incidence dir should be h = n so woh = woz
        const color  F = fresnel::conductor(cos_theta, eta, k);

        pdf              = 1.0f;
        reflected_colour = tint * F / cos_theta;

        return sd.shading_frame.to_world(wi_local);
    }
    //~ Walter eq
    //~ cos_h = sqrt((1 - xi1) / ((a^2 - 1)*xi1 + 1))
    //~ phi   = 2*pi*xi2

    //~ Beckmann sampling swapped to GGX for being heavy
    // const float tan2 = -a2 * std::log(1.0f - r1);
    // const float cos_h = 1.0f / std::sqrt(1.0f + tan2);

    // TODO: VNDF sampling Hetz - importance-sample D*G1 instead
    // of just D kills fireflies on grazing wo

    const float r1 = s->next();
    const float r2 = s->next();
    const float a2 = alpha * alpha;

    const float cos_theta_h = std::sqrt(std::max(0.0f,
                                (1.0f - r1) / ((a2 - 1.0f) * r1 + 1.0f)));

    const float sin_theta_h = std::sqrt(std::max(0.0f,
                                1.0f - cos_theta_h * cos_theta_h));

    const float phi = 2.0f * math::pi<float> * r2;

    const vec3 h_local(std::cos(phi) * sin_theta_h,
                       std::sin(phi) * sin_theta_h,
                       cos_theta_h);

    //~ wi = 2(wo.h)h - wo
    const float    wo_dot_h = math::dot(wo_local, h_local);
    const vec3 wi_local = (h_local * (2.0f * wo_dot_h)) - wo_local;

    if (wi_local.z <= 0.0f)
    {
        pdf = 0.0f;
        reflected_colour = color(0.0f, 0.0f, 0.0f);
        return sd.shading_frame.to_world(wi_local);
    }

    const float D = ggx::d(h_local, alpha);
    const float G = ggx::g(wi_local, wo_local, alpha);
    const color F = fresnel::conductor(std::fabs(wo_dot_h), eta, k);

    const float abs_wo_dot_h = std::max(
        math::epsilon<float>,
        std::fabs(wo_dot_h)
    );

    //~ half-vector PDF transform
    pdf = D * cos_theta_h / (4.0f * abs_wo_dot_h);

    const float denom = 4.0f * std::max(math::epsilon<float>, wo_local.z)
                       * std::max(math::epsilon<float>, wi_local.z);

    reflected_colour = tint * F * (D * G / denom);
    return sd.shading_frame.to_world(wi_local);
}

fox_tracer::color fox_tracer::bsdf::conductor::evaluate(
    const shading_data &sd, const vec3 &wi)
{
    if (alpha < math::epsilon<float>) return {0.0f, 0.0f, 0.0f};

    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);
    const vec3 wi_local = sd.shading_frame.to_local(wi);

    if (wi_local.z <= 0.0f || wo_local.z <= 0.0f)
        return {0.0f, 0.0f, 0.0f};


    // vec3 h_local = (wi_local + wo_local).normalize();
    vec3 h_local = (wi_local + wo_local);
    const float h_len = std::sqrt(h_local.x * h_local.x
                                + h_local.y * h_local.y
                                + h_local.z * h_local.z);

    if (h_len <= 0.0f) return {0.0f, 0.0f, 0.0f};

    h_local = h_local / h_len;

    //~ fr = F * D * G / (4 * cos_o * cos_i)
    const float D = ggx::d(h_local, alpha);
    const float G = ggx::g(wi_local, wo_local, alpha);
    const color F = fresnel::conductor(
        std::fabs(math::dot(wo_local, h_local)),
        eta, k);

    const color tint  = albedo->sample(sd.tu, sd.tv);
    const float denom = 4.0f * wo_local.z * wi_local.z;

    return tint * F * (D * G / denom);
}

float fox_tracer::bsdf::conductor::pdf(const shading_data &sd, const vec3 &wi)
{
    if (alpha < math::epsilon<float>) return 0.0f;

    //~ ref: p(wi) = D * cos_h / (4 * |wo.h|)

    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);
    const vec3 wi_local = sd.shading_frame.to_local(wi);

    if (wi_local.z <= 0.0f || wo_local.z <= 0.0f) return 0.0f;

    vec3 h_local = (wi_local + wo_local);
    const float h_len = std::sqrt(h_local.x * h_local.x
                                + h_local.y * h_local.y
                                + h_local.z * h_local.z);
    if (h_len <= 0.0f) return 0.0f;

    h_local = h_local / h_len;
    const float D = ggx::d(h_local, alpha);

    const float abs_wo_dot_h = std::max(math::epsilon<float>,
        std::fabs(math::dot(wo_local, h_local)));

    return D * std::fabs(h_local.z) / (4.0f * abs_wo_dot_h);
}

float fox_tracer::bsdf::conductor::mask(const shading_data &sd)
{
    //~ test opaque
    // return 1.0f;
    return albedo->sample_alpha(sd.tu, sd.tv);
}

bool fox_tracer::bsdf::conductor::is_pure_specular() const
{
    return alpha < math::epsilon<float>;
}

bool fox_tracer::bsdf::conductor::is_two_sided() const
{
    // test only: strict one-sided back faces black
    // return false;
    return true;
}

fox_tracer::bsdf::glass::glass(
    texture *_albedo, const float _int_ior,
    const float _ext_ior) noexcept
:   albedo(_albedo), int_ior(_int_ior),
    ext_ior(_ext_ior)
{}

fox_tracer::vec3 fox_tracer::bsdf::glass::sample(
    const shading_data &sd, sampler *s,
    color &reflected_colour,float &pdf)
{
    //~ Reference
    // smooth dielectric two Dirac lobes (reflect and refract)
    //~ P(reflect) = F, P(refract) = 1 - F
    //~ reflect wr = (-wo.x, -wo.y, wo.z)
    //~ refract Snell wt = -eta*wo + (eta*cos_i - cos_t) * n
    //~ where eta = eta_i / eta_t

    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);
    const bool  entering = wo_local.z > 0.0f;

    const float eta_i = entering ? ext_ior : int_ior;
    const float eta_t = entering ? int_ior : ext_ior;
    const float cos_theta_i = std::fabs(wo_local.z);

    const float F    = fresnel::dielectric(wo_local.z, int_ior, ext_ior);
    const color tint = albedo->sample(sd.tu, sd.tv);

    const float eta          = eta_i / eta_t;
    const float sin2_theta_t = eta * eta * std::max(0.0f,
                                   1.0f - cos_theta_i * cos_theta_i);

    if (const bool tir = (sin2_theta_t >= 1.0f); tir || s->next() < F)
    {
        //~ tir or russian roulette pick on F take the reflection lobe
        const vec3 wi_local(-wo_local.x, -wo_local.y, wo_local.z);
        const float cos_out = std::max(
            math::epsilon<float>,
            std::fabs(wi_local.z)
        );

        pdf = tir ? 1.0f : F;
        reflected_colour = tint * (pdf / cos_out);

        return sd.shading_frame.to_world(wi_local);
    }
    //~ refraction lobe
    const float cos_theta_t = std::sqrt(std::max(0.0f, 1.0f - sin2_theta_t));

    const vec3 wi_local(
        -eta * wo_local.x,
        -eta * wo_local.y,
        entering ? -cos_theta_t : cos_theta_t);

    const float cos_out = std::max(math::epsilon<float>, std::fabs(wi_local.z));
    const float one_minus_F = 1.0f - F;

    pdf = one_minus_F;

    const float eta_scale = (eta_t / eta_i) * (eta_t / eta_i);
    reflected_colour = tint * (one_minus_F * eta_scale / cos_out);

    // TODO: rough dielectric GGX d*g distribution on h if time
    // TODO: wave length dependent ior chromatic dispersion or prism if time
    // TODO: Beer Lambert absorption inside the medium if time

    return sd.shading_frame.to_world(wi_local);
}

fox_tracer::color fox_tracer::bsdf::glass::evaluate(const shading_data &sd, const vec3 &wi)
{
    return {0.0f, 0.0f, 0.0f};
}

float fox_tracer::bsdf::glass::pdf(const shading_data &sd, const vec3 &wi)
{
    return 0.0f;
}

float fox_tracer::bsdf::glass::mask(const shading_data &sd)
{
    return albedo->sample_alpha(sd.tu, sd.tv);
}

bool fox_tracer::bsdf::glass::is_pure_specular() const
{
    return true;
}

bool fox_tracer::bsdf::glass::is_two_sided() const
{
    return false;
}

fox_tracer::bsdf::dielectric::dielectric(
    texture *_albedo, const float _int_ior,
    const float _ext_ior, const float roughness) noexcept
:   albedo(_albedo), int_ior(_int_ior),
    ext_ior(_ext_ior), alpha(1.62142f * std::sqrt(roughness))
{
    // TODO: full rough dielectric ggx btdf
}

fox_tracer::vec3 fox_tracer::bsdf::dielectric::sample(
    const shading_data &sd, sampler *s,
    color &reflected_colour, float &pdf)
{
    //~ fallback
    //~ fr = rho / pi,  p(wi) = cos_theta / pi

    // const vec3 wo_local = sd.shading_frame.to_local(sd.wo);
    // const bool entering = wo_local.z > 0.0f;
    // const float eta_i = entering ? ext_ior : int_ior;
    // const float eta_t = entering ? int_ior : ext_ior;
    //
    // const float r1 = s->next();
    // const float r2 = s->next();
    // const float a2 = alpha * alpha;
    // const float cos_h = std::sqrt((1.0f - r1) / ((a2 - 1.0f) * r1 + 1.0f));
    // const float sin_h = std::sqrt(std::max(0.0f, 1.0f - cos_h*cos_h));
    // const float phi   = 2.0f * math::pi<float> * r2;
    // const vec3  h(std::cos(phi)*sin_h, std::sin(phi)*sin_h, cos_h);
    //
    // const float wo_dot_h = math::dot(wo_local, h);
    // const float F        = fresnel::dielectric(wo_dot_h, int_ior, ext_ior);
    //
    // vec3 wi_local;
    // if (s->next() < F)
    // {
    //     wi_local = (h * (2.0f * wo_dot_h)) - wo_local;     // reflect
    // }
    // else
    // {
    //     const float eta = eta_i / eta_t;
    //     const float c   = wo_dot_h;
    //     const float k_  = 1.0f - eta*eta * (1.0f - c*c);
    //     wi_local = -wo_local * eta + h * (eta*c - std::sqrt(k_)); // refract
    // }

    // TODO: remove the cosine fallback
    // TODO: VNDF sampling
    vec3 wi = sampling::cosine_sample_hemisphere(s->next(), s->next());
    pdf = wi.z / math::pi<float>;

    reflected_colour = albedo->sample(sd.tu, sd.tv) / math::pi<float>;
    wi               = sd.shading_frame.to_world(wi);

    return wi;
}

fox_tracer::color fox_tracer::bsdf::dielectric::evaluate(
    const shading_data &sd, const vec3 &wi)
{
    return albedo->sample(sd.tu, sd.tv) / math::pi<float>;
}

float fox_tracer::bsdf::dielectric::pdf(const shading_data &sd, const vec3 &wi)
{
    const vec3 wi_local = sd.shading_frame.to_local(wi);
    return sampling::cosine_hemisphere_pdf(wi_local);
}

float fox_tracer::bsdf::dielectric::mask(const shading_data &sd)
{
    return albedo->sample_alpha(sd.tu, sd.tv);
}

bool fox_tracer::bsdf::dielectric::is_pure_specular() const
{
    return false;
}

bool fox_tracer::bsdf::dielectric::is_two_sided() const
{
    return false;
}

fox_tracer::bsdf::oren_nayar::oren_nayar(texture *_albedo, const float _sigma) noexcept
    : albedo(_albedo), sigma(_sigma)
{
    // TODO: precompute A, B
}

fox_tracer::vec3 fox_tracer::bsdf::oren_nayar::sample(
    const shading_data &sd, sampler *s,
    color &reflected_colour, float &pdf)
{
    //~ Oren-Nayar microfacet diffuse no closed form importance sampler
    //~ fall back to hemisphere sampling
    const auto mode = static_cast<hemisphere_sampling>(
               config().hemisphere_mode.load(std::memory_order_relaxed));

    vec3 wi_local;
    if (mode == hemisphere_sampling::uniform)
    {
        wi_local = sampling::uniform_sample_hemisphere(s->next(), s->next());
        pdf      = sampling::uniform_hemisphere_pdf(wi_local);
    }
    else
    {
        wi_local = sampling::cosine_sample_hemisphere(s->next(), s->next());
        pdf      = sampling::cosine_hemisphere_pdf(wi_local);
    }

    //~ to_world then evaluate does to_local again wasted transform
    // TODO: split out evaluate_local(wi_local) helper to skip the round-trip

    const vec3 wi_world = sd.shading_frame.to_world(wi_local);

    reflected_colour = evaluate(sd, wi_world);
    return wi_world;
}

fox_tracer::color fox_tracer::bsdf::oren_nayar::evaluate(
    const shading_data &sd, const vec3 &wi)
{
    //~ Oren-Nayar reference
    //~ f = (rho/pi) * (A + B * max(0, cos(phi_i - phi_o)) * sin(alpha) * tan(beta))
    //~ A = 1 - 0.5 * s2 / (s2 + 0.33)
    //~ B = 0.45 * s2 / (s2 + 0.09)
    //~ alpha = max(theta_i, theta_o),  beta = min(theta_i, theta_o)
    //~ s2 = sigma^2

    const vec3 wi_local = sd.shading_frame.to_local(wi);
    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);

    if (wi_local.z <= 0.0f || wo_local.z <= 0.0f)
        return {0.0f, 0.0f, 0.0f};

    // TODO: Fujii: try generalized ON - drops the max() and trig for faster
    const float s2 = sigma * sigma;
    const float A  = 1.0f - 0.5f * s2 / (s2 + 0.33f);
    const float B  = 0.45f * s2 / (s2 + 0.09f);

    const float sin_theta_i = std::sqrt(std::max(0.0f, 1.0f - wi_local.z * wi_local.z));
    const float sin_theta_o = std::sqrt(std::max(0.0f, 1.0f - wo_local.z * wo_local.z));

    float max_cos = 0.0f;

    if (sin_theta_i > math::epsilon<float> && sin_theta_o > math::epsilon<float>)
    {
        const float d_cos = (wi_local.x * wo_local.x + wi_local.y * wo_local.y)
                          / (sin_theta_i * sin_theta_o);
        max_cos = std::max(0.0f, d_cos);
    }

    //~ alpha = larger angle => smaller cos => smaller wi.z or wo.z
    //~ beta  = smaller angle => larger cos
    //~ sin(alpha) = sin of whichever has the smaller cos
    //~ tan(beta)  = sin/cos of whichever has the larger cos
    const float cos_alpha = std::min(wi_local.z, wo_local.z);
    const float cos_beta  = std::max(wi_local.z, wo_local.z);
    const float sin_alpha = std::sqrt(1 - cos_alpha*cos_alpha);
    const float tan_beta  = std::sqrt(1 - cos_beta*cos_beta) / cos_beta;

    const color rho = albedo->sample(sd.tu, sd.tv);
    return rho * (1.0 / math::pi<float>)
               * (A + B * max_cos * sin_alpha * tan_beta);
}

float fox_tracer::bsdf::oren_nayar::pdf(const shading_data &sd, const vec3 &wi)
{
    const auto mode = static_cast<hemisphere_sampling>(
        config().hemisphere_mode.load(std::memory_order_relaxed)
    );

    const vec3 wi_local = sd.shading_frame.to_local(wi);

    if (mode == hemisphere_sampling::uniform)
        return sampling::uniform_hemisphere_pdf(wi_local);

    return sampling::cosine_hemisphere_pdf(wi_local);
}

float fox_tracer::bsdf::oren_nayar::mask(const shading_data &sd)
{
    return albedo->sample_alpha(sd.tu, sd.tv);
}

bool fox_tracer::bsdf::oren_nayar::is_pure_specular() const
{
    return false;
}

bool fox_tracer::bsdf::oren_nayar::is_two_sided() const
{
    return true;
}

fox_tracer::bsdf::plastic::plastic(
    texture *_albedo, const float _int_ior,
    const float _ext_ior, const float roughness) noexcept
:   albedo(_albedo), int_ior(_int_ior),
    ext_ior(_ext_ior)
{
    const auto vv = std::max(0.0f, std::min(roughness, 1.0f));
    // alpha = vv * vv;
    alpha = std::max(0.001f, 1.62142f * std::sqrt(vv));
}

float fox_tracer::bsdf::plastic::alpha_to_phong_exponent() const noexcept
{
    return (2.0f / math::squared(std::max(alpha, 0.001f))) - 2.0f;
}

fox_tracer::vec3 fox_tracer::bsdf::plastic::sample(
    const shading_data &sd, sampler *s,
    color &reflected_colour, float &pdf)
{
    //~ Reference GGX specular coat + Lambertian substrate
    //~ fr = f * d*g / (4 coso cosi) + (1-f) * rho/pi
    //~ Fresnel weighted lobe pick at wo.n
    //~ p(spec) = f(wo.z),  o(diff) = 1 - f(wo.z)
    //~ p(wi) = F * p_spec + (1-f) * p_diff

    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);
    if (wo_local.z <= 0.0f)
    {
        pdf = 0.0f;
        reflected_colour = color(0.0f, 0.0f, 0.0f);
        return sd.shading_frame.to_world(vec3(0.0f, 0.0f, 1.0f));
    }

    const float F_pick = fresnel::dielectric(wo_local.z, int_ior, ext_ior);
    const color rho    = albedo->sample(sd.tu, sd.tv);

    // const bool pick_spec = s->next() < 0.5f;
    // pdf = 0.5f * pdf_spec + 0.5f * pdf_diff;

    vec3 wi_local;
    if (s->next() < F_pick)
    {
        //~ cos_h = sqrt((1 - xi1) / ((a^2 - 1)*xi1 + 1))
        //~ phi   = 2*pi*xi2
        //~ wi    = 2(wo.h)h - wo

        const float r1 = s->next();
        const float r2 = s->next();
        const float a2 = alpha * alpha;

        const float cos_theta_h = std::sqrt(std::max(0.0f,
                                        (1.0f - r1) / ((a2 - 1.0f) * r1 + 1.0f)));

        const float sin_theta_h = std::sqrt(std::max(0.0f,
                                        1.0f - cos_theta_h * cos_theta_h));
        const float phi = 2.0f * math::pi<float> * r2;

        const vec3 h_local(std::cos(phi) * sin_theta_h,
                               std::sin(phi) * sin_theta_h,
                               cos_theta_h);

        const float wo_dot_h = math::dot(wo_local, h_local);

        wi_local = (h_local * (2.0f * wo_dot_h)) - wo_local;
        // TODO: VNDF sampling Heitz - lower variance on grazing wo
    }
    else
    {
        wi_local = sampling::cosine_sample_hemisphere(s->next(), s->next());
    }

    if (wi_local.z <= 0.0f)
    {
        pdf = 0.0f;
        reflected_colour = color(0.0f, 0.0f, 0.0f);
        return sd.shading_frame.to_world(wi_local);
    }

    // vec3 h_local = (wi_local + wo_local).normalize();
    vec3 h_local = wi_local + wo_local;
    const float h_len = std::sqrt(h_local.x * h_local.x
                                + h_local.y * h_local.y
                                + h_local.z * h_local.z);
    if (h_len <= 0.0f)
    {
        pdf = 0.0f;
        reflected_colour = color(0.0f, 0.0f, 0.0f);
        return sd.shading_frame.to_world(wi_local);
    }
    h_local = h_local / h_len;

    //~ spec = rho * F * D * G / (4 cos_o cos_i)
    //~ diff = rho * (1-F) * 1/pi
    const float D  = ggx::d(h_local, alpha);
    const float G  = ggx::g(wi_local, wo_local, alpha);
    const float wo_dot_h = std::max(math::epsilon<float>,
                               std::fabs(math::dot(wo_local, h_local)));
    const float F  = fresnel::dielectric(wo_dot_h, int_ior, ext_ior);

    const float denom_spec = 4.0f * std::max(math::epsilon<float>, wo_local.z)
                                  * std::max(math::epsilon<float>, wi_local.z);

    const color f_spec = rho * F          * (D * G / denom_spec);
    const color f_diff = rho * (1.0f - F) * (1.0f / math::pi<float>);

    // TODO: try out Kulla-Conty energy preserving fix - rho should not
    // appear on spec multi scattering compensation needed
    const float pdf_spec = D * std::fabs(h_local.z) / (4.0f * wo_dot_h);
    const float pdf_diff = wi_local.z / math::pi<float>;

    //~ MIS combined PDF F_pick weight matches the lobe pick probability
    pdf = F_pick * pdf_spec + (1.0f - F_pick) * pdf_diff;
    if (pdf <= 0.0f || !std::isfinite(pdf))
    {
        pdf = 0.0f;
        reflected_colour = color(0.0f, 0.0f, 0.0f);
        return sd.shading_frame.to_world(wi_local);
    }

    reflected_colour = f_spec + f_diff;
    return sd.shading_frame.to_world(wi_local);
}

fox_tracer::color fox_tracer::bsdf::plastic::evaluate(
    const shading_data &sd, const vec3 &wi)
{
    //~ reference: fr = F * D*G / (4 cos_o cos_i)  +  (1-F) * rho/pi
    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);
    const vec3 wi_local = sd.shading_frame.to_local(wi);

    if (wi_local.z <= 0.0f || wo_local.z <= 0.0f)
        return {0.0f, 0.0f, 0.0f};

    vec3 h_local = wi_local + wo_local;
    const float h_len = std::sqrt(h_local.x * h_local.x
                                + h_local.y * h_local.y
                                + h_local.z * h_local.z);
    if (h_len <= 0.0f) return {0.0f, 0.0f, 0.0f};
    h_local = h_local / h_len;

    const float D  = ggx::d(h_local, alpha);
    const float G  = ggx::g(wi_local, wo_local, alpha);
    const float wo_dot_h = std::max(math::epsilon<float>,
                               std::fabs(math::dot(wo_local, h_local)));
    const float F  = fresnel::dielectric(wo_dot_h, int_ior, ext_ior);

    const color rho = albedo->sample(sd.tu, sd.tv);
    const float denom_spec = 4.0f * wo_local.z * wi_local.z;

    const color f_spec = rho * F          * (D * G / denom_spec);
    const color f_diff = rho * (1.0f - F) * (1.0f / math::pi<float>);
    return f_spec + f_diff;
}

float fox_tracer::bsdf::plastic::pdf(const shading_data &sd, const vec3 &wi)
{
    //~ refernce: fpick * pspec + (1-fpick) * p_diff
    const vec3 wo_local = sd.shading_frame.to_local(sd.wo);
    const vec3 wi_local = sd.shading_frame.to_local(wi);

    if (wi_local.z <= 0.0f || wo_local.z <= 0.0f) return 0.0f;

    vec3 h_local = wi_local + wo_local;
    const float h_len = std::sqrt(h_local.x * h_local.x
                                + h_local.y * h_local.y
                                + h_local.z * h_local.z);
    if (h_len <= 0.0f) return 0.0f;
    h_local = h_local / h_len;

    const float D = ggx::d(h_local, alpha);
    const float wo_dot_h = std::max(math::epsilon<float>,
                               std::fabs(math::dot(wo_local, h_local)));
    const float F_pick = fresnel::dielectric(wo_local.z, int_ior, ext_ior);

    const float pdf_spec = D * std::fabs(h_local.z) / (4.0f * wo_dot_h);
    const float pdf_diff = wi_local.z / math::pi<float>;
    return F_pick * pdf_spec + (1.0f - F_pick) * pdf_diff;
}

float fox_tracer::bsdf::plastic::mask(const shading_data &sd)
{
    return albedo->sample_alpha(sd.tu, sd.tv);
}

bool fox_tracer::bsdf::plastic::is_pure_specular() const
{
    return false;
}

bool fox_tracer::bsdf::plastic::is_two_sided() const
{
    return true;
}

fox_tracer::bsdf::layered::layered(
    base *_base, const color &_sigma_a,
    const float _thickness, const float _int_ior,
    const float _ext_ior) noexcept
    :   substrate(_base), sigma_a(_sigma_a),
        thickness(_thickness), int_ior(_int_ior),
        ext_ior(_ext_ior)
{
    // TODO: full coated dielectric
}

fox_tracer::vec3 fox_tracer::bsdf::layered::sample(
    const shading_data &sd, sampler *s,
    color &reflected_colour, float &pdf)
{
    // TODO: position-free Monte Carlo for arbitrary stacks
    // TODO: precomputed albedo tables for real-time eval
    return substrate->sample(sd, s, reflected_colour, pdf);
}

fox_tracer::color fox_tracer::bsdf::layered::evaluate(
    const shading_data &sd, const vec3 &wi)
{
    // TODO: fr = fr_top(wo, wi) + T(wo) * absorb * fr_sub(wo', wi') * T(wi)
    //   with wo'/wi' the refracted directions inside the medium
    return substrate->evaluate(sd, wi);
}

float fox_tracer::bsdf::layered::pdf(const shading_data &sd, const vec3 &wi)
{
    // TODO: F in * pdf top + (1 - Fin) * pdf sub refracted
    return substrate->pdf(sd, wi);
}

float fox_tracer::bsdf::layered::mask(const shading_data &sd)
{
    return substrate->mask(sd);
}

bool fox_tracer::bsdf::layered::is_pure_specular() const
{
   return substrate->is_pure_specular();
}

bool fox_tracer::bsdf::layered::is_two_sided() const
{
    return substrate->is_two_sided();
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
