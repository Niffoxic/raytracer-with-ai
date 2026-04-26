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
#include "framework/lights.h"
#include "framework/geometry.h"
#include "framework/imaging.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cmath>

#include "framework/materials.h"

fox_tracer::vec3 fox_tracer::lights::area::
sample(const shading_data &sd, sampler *s, color &emitted_colour, float &pdf)
{
    // const int v = static_cast<int>(s->next() * 3.0f);
    // pdf = 1.0f;
    // emitted_colour = emission;
    // return tri->vertices[v].position;

    emitted_colour = emission;
    return tri->sample(s, pdf);
}

fox_tracer::color fox_tracer::lights::area::evaluate(const vec3 &wi)
{
    if (math::dot(wi, tri->g_normal()) < 0.0f)
    {
        return emission;
    }
    return {0.0f, 0.0f, 0.0f};
}

float fox_tracer::lights::area::pdf(const shading_data &sd, const vec3 &wi)
{
    // const vec3 to_light = tri->centre() - sd.x;
    // const float d2 = math::dot(to_light, to_light);
    // const float cos_l = std::max(0.0f, math::dot(-wi, tri->g_normal()));
    // return d2 / (cos_l * tri->area);

    return 1.0f / tri->area;
}

bool fox_tracer::lights::area::is_area() const
{
    return true;
}

float fox_tracer::lights::area::total_integrated_power()
{
    return tri->area * emission.luminance();
}

fox_tracer::vec3 fox_tracer::lights::area::normal(
    const shading_data &sd, const vec3 &wi)
{
    return tri->g_normal();
}

fox_tracer::vec3 fox_tracer::lights::area::sample_position_from_light(
    sampler *s, float &pdf)
{
    //~ p(x) = 1 / A
    return tri->sample(s, pdf);
}

fox_tracer::vec3 fox_tracer::lights::area::sample_direction_from_light(
    sampler *s, float &pdf)
{
    const vec3 wi_local = sampling::cosine_sample_hemisphere(
        s->next(), s->next());
    pdf = sampling::cosine_hemisphere_pdf(wi_local);

    // const vec3 wi_local = sampling::uniform_sample_hemisphere(s->next(), s->next());
    // pdf = sampling::uniform_hemisphere_pdf(wi_local);

    frame fr;
    fr.from_vector(tri->g_normal());

    return fr.to_world(wi_local);
}

fox_tracer::lights::background_colour::background_colour(
    const color &_emission) noexcept
    : emission(_emission)
{}

fox_tracer::vec3 fox_tracer::lights::background_colour::sample(
    const shading_data &sd, sampler *s,
    color &emitted_colour, float &pdf)
{
    //~ test: sky lighting
    // vec3 wi = sampling::uniform_sample_hemisphere(s->next(), s->next());
    // pdf = sampling::uniform_hemisphere_pdf(wi);
    // wi  = sd.shading_frame.to_world(wi);
    // emitted_colour = emission;
    // return wi;

    const vec3 wi = sampling::uniform_sample_sphere(s->next(), s->next());
    pdf = sampling::uniform_sphere_pdf(wi);
    emitted_colour = emission;
    return wi;
}

fox_tracer::color fox_tracer::lights::background_colour::evaluate(const vec3 &wi)
{
    return emission;
}

float fox_tracer::lights::background_colour::pdf(
    const shading_data &sd, const vec3 &wi)
{
    return sampling::uniform_sphere_pdf(wi);
}

bool fox_tracer::lights::background_colour::is_area() const
{
    return false;
}

float fox_tracer::lights::background_colour::total_integrated_power()
{
    return emission.luminance() * 4.0f * math::pi<float>;
}

fox_tracer::vec3 fox_tracer::lights::background_colour::normal(
    const shading_data &sd, const vec3 &wi)
{
    return -wi;
}

fox_tracer::vec3 fox_tracer::lights::background_colour::sample_position_from_light(
    sampler *s, float &pdf)
{
    vec3 p = sampling::uniform_sample_sphere(s->next(), s->next());

    const auto&[scene_centre, scene_radius] =
        singleton::use<scene_bounds>();
    p = p * scene_radius;
    p = p + scene_centre;

    // pdf = 1.0f / (4.0f * math::pi<float>);

    pdf = 4.0f * math::pi<float> * scene_radius * scene_radius;
    return p;
}

fox_tracer::vec3 fox_tracer::lights::background_colour::sample_direction_from_light(
    sampler *s, float &pdf)
{
    const vec3 wi   = sampling::uniform_sample_sphere(s->next(), s->next());
    pdf             = sampling::uniform_sphere_pdf(wi);
    return wi;
}

fox_tracer::lights::environment_map::environment_map(texture *_env) noexcept
    : env(_env)
{
    // TODO: build 2D inverse Ccdf over luminance for importance sampling
}

fox_tracer::vec3 fox_tracer::lights::environment_map::sample(
    const shading_data &sd, sampler *s,
    color &emitted_colour, float &pdf)
{
    const vec3 wi  = sampling::uniform_sample_sphere(s->next(), s->next());
    // pdf         = sampling::cosine_hemisphere_pdf(wi_local);
    pdf            = sampling::uniform_sphere_pdf(wi);
    emitted_colour = evaluate(wi);

    return wi;
}

fox_tracer::color fox_tracer::lights::environment_map::evaluate(const vec3 &wi)
{
    float u = std::atan2(wi.z, wi.x);
    u = (u < 0.0f) ? u + (2.0f * math::pi<float>) : u;
    u = u / (2.0f * math::pi<float>);

    const float v = std::acos(wi.y) / math::pi<float>;
    return env->sample(u, v);
}

float fox_tracer::lights::environment_map::pdf(const shading_data &sd, const vec3 &wi)
{
    return sampling::uniform_sphere_pdf(wi);
}

bool fox_tracer::lights::environment_map::is_area() const
{
    return false;
}

float fox_tracer::lights::environment_map::total_integrated_power()
{
    // TODO: cache result only recompute when env texture changes

    float total = 0.0f;
    for (int i = 0; i < env->height; ++i)
    {
        // const float st = std::sin(((static_cast<float>(i) + 0.5f)
        //                          / static_cast<float>(env->height))
        //                         * math::pi<float>);

        const float st = std::sin((static_cast<float>(i)
                                   / static_cast<float>(env->height))
                                  * math::pi<float>);

        for (int n = 0; n < env->width; ++n)
        {
            total += env->texels[(i * env->width) + n].luminance() * st;
        }
    }
    total = total / static_cast<float>(env->width * env->height);
    return total * 4.0f * math::pi<float>;
}

fox_tracer::vec3 fox_tracer::lights::environment_map::normal(const shading_data &sd, const vec3 &wi)
{
    return -wi;
}

fox_tracer::vec3 fox_tracer::lights::environment_map::sample_position_from_light(sampler *s, float &pdf)
{
    vec3 p = sampling::uniform_sample_sphere(s->next(), s->next());
    const auto&[scene_centre, scene_radius] =
        singleton::use<scene_bounds>();
    p = p * scene_radius;
    p = p + scene_centre;

    pdf = 1.0f / (4.0f * math::pi<float> * math::squared(scene_radius));
    return p;
}

fox_tracer::vec3 fox_tracer::lights::environment_map::sample_direction_from_light(sampler *s, float &pdf)
{
    const vec3 wi = sampling::uniform_sample_sphere(s->next(), s->next());
    pdf = sampling::uniform_sphere_pdf(wi);
    return wi;
}
