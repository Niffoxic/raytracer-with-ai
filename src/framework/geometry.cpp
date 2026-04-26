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
#include "framework/geometry.h"

#include "sampler/sampling.h"

fox_tracer::geometry::ray::ray(const vec3 &_o, const vec3 &_d) noexcept
{
    init(_o, _d);
}

void fox_tracer::geometry::ray::init(const vec3 &_o, const vec3 &_d) noexcept
{
    o           = _o;
    dir         = _d;
    inv_dir     = vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    o.w         = 0.0f;
    dir.w       = 0.0f;
    inv_dir.w   = 0.0f;
}

fox_tracer::vec3 fox_tracer::geometry::ray::at(const float t) const noexcept
{
    return o + (dir * t);
}

void fox_tracer::geometry::plane::init(const vec3 &_n, const float _d) noexcept
{
    n = _n;
    d = _d;
}

bool fox_tracer::geometry::plane::ray_intersect(const ray &r, float &t) const noexcept
{
    const float denom = math::dot(n, r.dir);

    if (std::fabs(denom) < math::epsilon<float>)
    {
        return false;
    }

    t = -(math::dot(n, r.o) + d) / denom;
    return t >= 0.0f;
}

void fox_tracer::geometry::triangle::init(
    const vertex &v0, const vertex &v1,
    const vertex &v2, const unsigned int _material_index) noexcept
{
    material_index = _material_index;

    vertices[0]    = v0;
    vertices[1]    = v1;
    vertices[2]    = v2;

    e1   = vertices[2].position - vertices[1].position;
    e2   = vertices[0].position - vertices[2].position;
    e0p  = vertices[1].position - vertices[0].position;
    e1p  = vertices[2].position - vertices[0].position;

    n    = e1.cross(e2).normalize();
    area = e1.cross(e2).length() * 0.5f;
    d    = math::dot(n, vertices[0].position);
}

fox_tracer::vec3 fox_tracer::geometry::triangle::centre() const noexcept
{
    return (vertices[0].position + vertices[1].position + vertices[2].position) / 3.0f;
}

bool fox_tracer::geometry::triangle::ray_intersect(
    const ray &r, float &t,
    float &u, float &v) const noexcept
{
    const vec3 p    = r.dir.cross(e1p);
    const float det = e0p.dot(p);

    if (std::fabs(det) < math::epsilon<float>)
    {
        return false;
    }

    const float inv_det = 1.0f / det;
    const vec3 T        = r.o - vertices[0].position;
    const float a       = T.dot(p) * inv_det;

    if (a < 0.0f || a > 1.0f)
    {
        return false;
    }

    const vec3  q = T.cross(e0p);
    const float b = r.dir.dot(q) * inv_det;

    if (b < 0.0f || (a + b) > 1.0f)
    {
        return false;
    }

    const float t_hit = e1p.dot(q) * inv_det;

    if (t_hit < 0.0f)
    {
        return false;
    }

    t = t_hit;
    u = 1.0f - a - b;
    v = a;
    return true;
}

void fox_tracer::geometry::triangle::interpolate_attributes(
    const float alpha, const float beta, const float gamma,
    vec3 &interpolated_normal, float &interpolated_u,
    float &interpolated_v) const noexcept
{
    interpolated_normal = vertices[0].normal * alpha
                        + vertices[1].normal * beta
                        + vertices[2].normal * gamma;

    interpolated_normal = interpolated_normal.normalize();

    interpolated_u = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
    interpolated_v = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
}

fox_tracer::vec3 fox_tracer::geometry::triangle::sample(sampler *s, float &pdf) const noexcept
{
    const float r1 = s->next();
    const float r2 = s->next();

    const float su = std::sqrt(r1);
    const float u  = 1.0f - su;
    const float v  = r2 * su;
    const float w  = 1.0f - u - v;

    pdf = (area > 0.0f) ? (1.0f / area) : 0.0f;

    return vertices[0].position * u + vertices[1].position * v + vertices[2].position * w;
}

fox_tracer::vec3 fox_tracer::geometry::triangle::g_normal() const noexcept
{
    return (n * (math::dot(vertices[0].normal, n) > 0.0f ? 1.0f : -1.0f));

}

fox_tracer::geometry::aabb::aabb() noexcept
{
    reset();
}

void fox_tracer::geometry::aabb::reset() noexcept
{
    max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    min = vec3( FLT_MAX,  FLT_MAX,  FLT_MAX);
}

void fox_tracer::geometry::aabb::extend(const vec3 &p) noexcept
{
    max = math::max(max, p);
    min = math::min(min, p);
}

bool fox_tracer::geometry::aabb::ray_aabb(const ray &r, float &t) const noexcept
{
    const __m128 origin = _mm_load_ps(r.o.data);
    const __m128 invd   = _mm_load_ps(r.inv_dir.data);
    const __m128 mn     = _mm_load_ps(min.data);
    const __m128 mx     = _mm_load_ps(max.data);

    const __m128 t0 = _mm_mul_ps(_mm_sub_ps(mn, origin), invd);
    const __m128 t1 = _mm_mul_ps(_mm_sub_ps(mx, origin), invd);

    const __m128 tmn = _mm_min_ps(t0, t1);
    const __m128 tmx = _mm_max_ps(t0, t1);

    alignas(16) float mn_arr[4];
    alignas(16) float mx_arr[4];

    _mm_store_ps(mn_arr, tmn);
    _mm_store_ps(mx_arr, tmx);

    const float t_entry = std::max(std::max(mn_arr[0], mn_arr[1]), mn_arr[2]);
    const float t_exit  = std::min(std::min(mx_arr[0], mx_arr[1]), mx_arr[2]);

    if (t_entry <= t_exit && t_exit >= 0.0f)
    {
        t = (t_entry > 0.0f) ? t_entry : 0.0f;
        return true;
    }
    return false;
}

bool fox_tracer::geometry::aabb::ray_aabb(const ray &r) const noexcept
{
    float t;
    return ray_aabb(r, t);
}

float fox_tracer::geometry::aabb::area() const noexcept
{
    const vec3 size = max - min;
    return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
}

void fox_tracer::geometry::sphere::init(const vec3 &_centre, float _radius) noexcept
{
    centre = _centre;
    radius = _radius;
}

bool fox_tracer::geometry::sphere::ray_intersect(const ray &r, float &t) const noexcept
{
    const vec3  h    = r.o - centre;
    const float b    = math::dot(h, r.dir);
    const float c    = math::dot(h, h) - radius * radius;
    const float disc = b * b - c;

    if (disc < 0.0f)
    {
        return false;
    }

    const float sqrt_disc = std::sqrt(disc);
    const float t0 = -b - sqrt_disc;
    const float t1 = -b + sqrt_disc;

    if (t0 > 0.0f)
    {
        t = t0;
        return true;
    }

    if (t1 > 0.0f)
    {
        t = t1;
        return true;
    }

    return false;
}
