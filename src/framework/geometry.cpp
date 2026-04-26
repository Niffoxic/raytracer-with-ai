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

fox_tracer::accelerated_structure::bvh_node::~bvh_node()
{
    delete l;
    delete r;
}

void fox_tracer::accelerated_structure::bvh_node::build(
    std::vector<triangle> &input_triangles,
    std::vector<triangle> &output_triangles)
{
    output_triangles.reserve(output_triangles.size() + input_triangles.size());

    bounds.reset();

    // for (const auto& t: input_triangles) bounds.extend(t.centre());

    // aabb tri_b;
    // for (const auto& t: input_triangles)
    // {
    //     tri_b.reset();
    //     tri_b.extend(t.vertices[0].position);
    //     tri_b.extend(t.vertices[1].position);
    //     tri_b.extend(t.vertices[2].position);
    //     bounds.extend(tri_b.min);
    //     bounds.extend(tri_b.max);
    // }

    for (const auto & input_triangle : input_triangles)
    {
        bounds.extend(input_triangle.vertices[0].position);
        bounds.extend(input_triangle.vertices[1].position);
        bounds.extend(input_triangle.vertices[2].position);
    }

    build_recursive(input_triangles, output_triangles);
}

void fox_tracer::accelerated_structure::bvh_node::traverse(
    const ray &r_, const std::vector<triangle> &scene_triangles,
    intersection_data &intersection) const
{
    // TODO: iterative traversal with explicit stack for no recursion overhead
    // TODO: ray packets / SIMD stest (4 or 8 rays at once)

    float t_bounds;
    if (!bounds.ray_aabb(r_, t_bounds)) return;
    if (t_bounds >= intersection.t) return;

    if (num > 0)
    {
        for (unsigned int i = 0; i < num; ++i)
        {
            float u, v;

            // for (unsigned int i = 0; i < num; ++i)
            // {
            //     float u, v, t;
            //     if (scene_triangles[offset + i].ray_intersect(r_, t, u, v))
            //     {
            //         if (t < intersection.t)
            //         {
            //             intersection.t     = t;
            //             intersection.ID    = offset + i;
            //             intersection.alpha = u;
            //             intersection.beta  = v;
            //             intersection.gamma = 1.0f - (u + v);
            //         }
            //     }
            // }

            if (float t; triangles[i].ray_intersect(r_, t, u, v))
            {
                if (t < intersection.t)
                {
                    intersection.t     = t;
                    intersection.ID    = offset + i;
                    intersection.alpha = u;
                    intersection.beta  = v;
                    intersection.gamma = 1.0f - (u + v);
                }
            }
        }

        return;
    }

    //~ visit nearer child first prune farther on t_far >= current best t
    if (l && r)
    {
        float t_l = FLT_MAX;
        float t_r = FLT_MAX;
        const bool hit_l = l->bounds.ray_aabb(r_, t_l);
        const bool hit_r = r->bounds.ray_aabb(r_, t_r);

        // if (l) l->traverse(r_, scene_triangles, intersection);
        // if (r) r->traverse(r_, scene_triangles, intersection);

        // if (hit_l) l->traverse(r_, scene_triangles, intersection);
        // if (hit_r && t_r < intersection.t) r->traverse(r_, scene_triangles, intersection);

        if (hit_l && hit_r)
        {
            if (t_l <= t_r)
            {
                l->traverse(r_, scene_triangles, intersection);
                if (t_r < intersection.t) r->traverse(r_, scene_triangles, intersection);
            }
            else
            {
                r->traverse(r_, scene_triangles, intersection);
                if (t_l < intersection.t) l->traverse(r_, scene_triangles, intersection);
            }
        }
        else if (hit_l)
        {
            l->traverse(r_, scene_triangles, intersection);
        }
        else if (hit_r)
        {
            r->traverse(r_, scene_triangles, intersection);
        }
    }
    else if (l)
    {
        l->traverse(r_, scene_triangles, intersection);
    }
    else if (r)
    {
        r->traverse(r_, scene_triangles, intersection);
    }
}

fox_tracer::accelerated_structure::intersection_data
fox_tracer::accelerated_structure::bvh_node::traverse(
    const ray &r_,
    const std::vector<triangle> &scene_triangles) const
{
    intersection_data intersection;
    intersection.t = FLT_MAX;
    traverse(r_, scene_triangles, intersection);
    return intersection;
}

bool fox_tracer::accelerated_structure::bvh_node::traverse_visible(
    const ray &r_, const std::vector<triangle> &scene_triangles,
    const float max_t) const
{
    float t_bounds;
    if (!bounds.ray_aabb(r_, t_bounds)) return true;
    if (t_bounds >= max_t) return true;

    if (num > 0)
    {
        for (unsigned int i = 0; i < num; ++i)
        {
            float u, v;
            if (float t; triangles[i].ray_intersect(r_, t, u, v))
            {
                if (t < max_t) return false;
            }
        }
        return true;
    }

    // intersection_data hit;
    // hit.t = FLT_MAX;
    // traverse(r_, scene_triangles, hit);
    // return hit.t >= max_t;

    // float t_l = FLT_MAX, t_r = FLT_MAX;
    // const bool hit_l = l && l->bounds.ray_aabb(r_, t_l);
    // const bool hit_r = r && r->bounds.ray_aabb(r_, t_r);
    // if (hit_l && t_l <= t_r)
    // {
    //     if (!l->traverse_visible(r_, scene_triangles, max_t)) return false;
    //     if (hit_r) return r->traverse_visible(r_, scene_triangles, max_t);
    // }
    // else if (hit_r)
    // {
    //     if (!r->traverse_visible(r_, scene_triangles, max_t)) return false;
    //     if (hit_l) return l->traverse_visible(r_, scene_triangles, max_t);
    // }
    // return true;


    if (l && !l->traverse_visible(r_, scene_triangles, max_t)) return false;
    if (r && !r->traverse_visible(r_, scene_triangles, max_t)) return false;
    // TODO: alpha mask aware shadows call bsdf->mask at hit
    return true;
}

float fox_tracer::accelerated_structure::bvh_node::axis_of(
    const vec3 &v, const int axis) noexcept
{
    return (axis == 0) ? v.x : ((axis == 1) ? v.y : v.z);
}

void fox_tracer::accelerated_structure::bvh_node::build_recursive(
    std::vector<triangle> &node_tris,
    std::vector<triangle> &output_triangles)
{
    const auto N = static_cast<unsigned int>(node_tris.size());

    if (N <= config::max_node_triangles)
    {
        make_leaf(node_tris, output_triangles);
        return;
    }

    const float parent_area = bounds.area();
    const float leaf_cost   = static_cast<float>(N) * config::triangle_cost;

    int   best_axis    = -1;
    int   best_bin     = -1;
    float best_cost    = FLT_MAX;
    float best_min_c   = 0.0f;
    float best_range_c = 0.0f;

    //~ TODO: add this on imgui controls its still fast tho
    // const int axis = bounds.longest_axis();
    // std::ranges::sort(node_tris,
    //     [axis](const triangle& a, const triangle& b)
    //     {
    //         return axis_of(a.centre(), axis) < axis_of(b.centre(), axis);
    //     });
    // const size_t mid = node_tris.size() / 2;

    for (int axis = 0; axis < 3; ++axis)
    {
        //~ centroid range on this axis
        float min_c =  FLT_MAX;
        float max_c = -FLT_MAX;

        // auto [it_min, it_max] = std::minmax_element(node_tris.begin(), node_tris.end(),
        //     [axis](const triangle& a, const triangle& b)
        //     {
        //         return axis_of(a.centre(), axis) < axis_of(b.centre(), axis);
        //     });
        // min_c = axis_of(it_min->centre(), axis);
        // max_c = axis_of(it_max->centre(), axis);

        for (unsigned int i = 0; i < N; ++i)
        {
            const float c = axis_of(node_tris[i].centre(), axis);
            if (c < min_c) min_c = c;
            if (c > max_c) max_c = c;
        }

        const float range_c = max_c - min_c;
        if (range_c < math::epsilon<float>)
        {
            //~ no split possible because all centroids coincident on this axis
            continue;
        }

        aabb bin_bounds[config::build_bins];
        int  bin_count[config::build_bins];
        for (int & i : bin_count) i = 0;

        // std::array + std::fill version, identical codegen, dropped for terseness:
        // std::array<aabb, config::build_bins> bin_bounds{};
        // std::array<int, config::build_bins> bin_count{};
        // bin_count.fill(0);

        //~ bucket index
        const float inv_range = static_cast<float>(config::build_bins) / range_c;

        for (unsigned int i = 0; i < N; ++i)
        {
            const float c = axis_of(node_tris[i].centre(), axis);
            int b = static_cast<int>((c - min_c) * inv_range);

            b = std::clamp(b, 0, config::build_bins - 1);

            // aabb tb;
            // tb.extend(node_tris[i].vertices[0].position);
            // tb.extend(node_tris[i].vertices[1].position);
            // tb.extend(node_tris[i].vertices[2].position);
            // bin_bounds[b].extend(tb.min);
            // bin_bounds[b].extend(tb.max);

            bin_count[b]++;
            bin_bounds[b].extend(node_tris[i].vertices[0].position);
            bin_bounds[b].extend(node_tris[i].vertices[1].position);
            bin_bounds[b].extend(node_tris[i].vertices[2].position);
        }

        //~ left prefix sweep
        aabb left_accum[config::build_bins];
        int  left_count[config::build_bins];
        {
            aabb accum;
            int  cnt = 0;
            for (int i = 0; i < config::build_bins; ++i)
            {
                cnt += bin_count[i];
                if (bin_count[i] > 0)
                {
                    accum.extend(bin_bounds[i].min);
                    accum.extend(bin_bounds[i].max);
                }
                left_accum[i] = accum;
                left_count[i] = cnt;
            }
        }

        // go for the right suffix sweep
        aabb right_accum[config::build_bins];
        int  right_count[config::build_bins];
        {
            aabb accum;
            int  cnt = 0;
            for (int i = config::build_bins - 1; i >= 0; --i)
            {
                cnt += bin_count[i];
                if (bin_count[i] > 0)
                {
                    accum.extend(bin_bounds[i].min);
                    accum.extend(bin_bounds[i].max);
                }
                right_accum[i] = accum;
                right_count[i] = cnt;
            }
        }

        //~ cost = C_t + (A_L * N_L * C_i + A_R * N_R * C_i) / A_parent
        for (int i = 0; i < config::build_bins - 1; ++i)
        {
            const int n_l = left_count[i];
            const int n_r = right_count[i + 1];

            if (n_l == 0 || n_r == 0) continue;

            const float a_l = left_accum[i].area();
            const float a_r = right_accum[i + 1].area();

            // const float cost = config::traverse_cost
            //     + a_l * static_cast<float>(n_l) * config::triangle_cost
            //     + a_r * static_cast<float>(n_r) * config::triangle_cost;

            const float cost = config::traverse_cost
                + (a_l * static_cast<float>(n_l) * config::triangle_cost) / parent_area
                + (a_r * static_cast<float>(n_r) * config::triangle_cost) / parent_area;

            if (cost < best_cost)
            {
                best_cost    = cost;
                best_axis    = axis;
                best_bin     = i;
                best_min_c   = min_c;
                best_range_c = range_c;
            }
        }
    }

    //~ avoiding pathological deep leaves!!
    if (best_axis < 0 || best_cost >= leaf_cost)
    {
        if (N > static_cast<unsigned int>(config::max_node_triangles * 8))
        {
            force_median_split(node_tris, output_triangles);
            return;
        }
        make_leaf(node_tris, output_triangles);
        return;
    }

    const float split_pos = best_min_c
                          + (static_cast<float>(best_bin + 1)
                             * best_range_c / static_cast<float>(config::build_bins));

    std::vector<triangle> left_tris;
    std::vector<triangle> right_tris;
    left_tris.reserve(N / 2);
    right_tris.reserve(N / 2);

    aabb left_bounds;
    aabb right_bounds;

    // auto mid = std::partition(node_tris.begin(), node_tris.end(),
    //     [&](const triangle& t)
    //     {
    //          return axis_of(t.centre(), best_axis) < split_pos;
    //     });

    for (unsigned int i = 0; i < N; ++i)
    {
        if (const float c = axis_of(node_tris[i].centre(), best_axis); c < split_pos)
        {
            left_tris.push_back(node_tris[i]);
            left_bounds.extend(node_tris[i].vertices[0].position);
            left_bounds.extend(node_tris[i].vertices[1].position);
            left_bounds.extend(node_tris[i].vertices[2].position);
        }
        else
        {
            right_tris.push_back(node_tris[i]);
            right_bounds.extend(node_tris[i].vertices[0].position);
            right_bounds.extend(node_tris[i].vertices[1].position);
            right_bounds.extend(node_tris[i].vertices[2].position);
        }
    }
    //~ all triangles on one side
    if (left_tris.empty() || right_tris.empty())
    {
        make_leaf(node_tris, output_triangles);
        return;
    }

    node_tris.clear();
    node_tris.shrink_to_fit();

    l = new bvh_node();
    l->bounds = left_bounds;
    l->build_recursive(left_tris, output_triangles);

    r = new bvh_node();
    r->bounds = right_bounds;
    r->build_recursive(right_tris, output_triangles);

    // TODO: parallel build using tbb or std::execution - subtree builds are independent
    // TODO: can try spatial split for high overlap scenes (future stuff I dont find complex scene anyways)
}

void fox_tracer::accelerated_structure::bvh_node::force_median_split(
    std::vector<triangle> &node_tris,
    std::vector<triangle> &output_triangles)
{
    const vec3 size = bounds.max - bounds.min;
    int axis = 0;
    if (size.y > size.x) axis = 1;
    if (size.z > ((axis == 1) ? size.y : size.x)) axis = 2;

    // std::nth_element(node_tris.begin(), node_tris.begin() + node_tris.size()/2,
    //                  node_tris.end(), cmp);

    std::ranges::sort(node_tris,
      [axis](const triangle& a, const triangle& b)
      {
          return axis_of(a.centre(), axis) < axis_of(b.centre(), axis);
      });

    const size_t mid = node_tris.size() / 2;
    std::vector left_tris (node_tris.begin(), node_tris.begin() + static_cast<long long>(mid));
    std::vector right_tris(node_tris.begin() +
        static_cast<long long>(mid), node_tris.end());

    aabb left_bounds;
    for (const auto & left_tri : left_tris)
    {
        left_bounds.extend(left_tri.vertices[0].position);
        left_bounds.extend(left_tri.vertices[1].position);
        left_bounds.extend(left_tri.vertices[2].position);
    }

    aabb right_bounds;
    for (const auto & right_tri : right_tris)
    {
        right_bounds.extend(right_tri.vertices[0].position);
        right_bounds.extend(right_tri.vertices[1].position);
        right_bounds.extend(right_tri.vertices[2].position);
    }

    node_tris.clear();
    node_tris.shrink_to_fit();

    l = new bvh_node();
    l->bounds = left_bounds;
    l->build_recursive(left_tris, output_triangles);

    r = new bvh_node();
    r->bounds = right_bounds;
    r->build_recursive(right_tris, output_triangles);
}

void fox_tracer::accelerated_structure::bvh_node::make_leaf(
    const std::vector<triangle> &node_tris,
    std::vector<triangle> &output_triangles)
{
    offset = static_cast<unsigned int>(output_triangles.size());
    num    = static_cast<unsigned int>(node_tris.size());

    for (unsigned int i = 0; i < num; ++i)
    {
        output_triangles.push_back(node_tris[i]);
    }
    triangles = &output_triangles[offset];

    // TODO: drop the triangles* cache index using output_triangles[offset+i]
    // at traversal time removes the realloc invalidation
}
