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
#include "scene/scene.h"
#include "scene/scene_loader.h"

#include "utils/assets_loader.h"
#include "sampler/sampling.h"
#include "config.h"

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "framework/imaging.h"
#include "utils/logger.h"

void fox_tracer::camera::init(
    const matrix &projection, const int screen_width,
    const int screen_height)
{
    projection_matrix         = projection;
    inverse_projection_matrix = projection.invert();
    width  = static_cast<float>(screen_width);
    height = static_cast<float>(screen_height);

    const float w_lens = 2.0f / projection.a[1][1];
    const float aspect = projection.a[0][0] / projection.a[1][1];
    const float h_lens = w_lens * aspect;
    a_film = std::fabs(w_lens * h_lens);

    const float t = projection.a[1][1];
    if (t > 0.0f)
    {
        // fov_deg = 2.0f * std::asin(1.0f / std::sqrt(1.0f + math::squared(t))) * (180.0f / math::pi<float>);
        // fov_deg = 2.0f * std::acos(t / std::sqrt(1.0f + math::squared(t))) * (180.0f / math::pi<float>);
        // fov_deg = 2.0f * std::atan2(1.0f, t) * (180.0f / math::pi<float>);
        fov_deg = 2.0f * std::atan(1.0f / t) * (180.0f / math::pi<float>);
    }
    world_to_clip = projection_matrix;
    flip_x = (projection.a[0][0] < 0.0f);
}

void fox_tracer::camera::update_view(const matrix &V)
{
    camera_to_world = V;
    camera_to_view  = V.invert();
    origin          = camera_to_world.mul_point(vec3(0.0f, 0.0f, 0.0f));

    view_direction = camera_to_world.mul_vec(vec3(0.0f, 0.0f, 1.0f)).normalize();

    // view_direction = inverse_projection_matrix
    //                     .mul_point_and_perspective_divide(vec3(0,0,1));
    // view_direction = camera_to_world.mul_vec(view_direction).normalize();
    world_to_clip = projection_matrix * camera_to_view;
    update_basis_cache();
}

void fox_tracer::camera::set_thin_lens(const float r_lens, const float f_dist) noexcept
{
    lens_radius    = r_lens;
    focal_distance = f_dist;
}

void fox_tracer::camera::apply_intrinsics(const float new_fov_deg) noexcept
{
    const float clamped_fov = std::clamp(new_fov_deg, 1.0f, 179.0f);
    if (std::fabs(clamped_fov - fov_deg) < 1.0e-4f) return;

    // const float aspect = (height > 0.0f)
    //                         ? (width / height)
    //                         : 1.0f;
    //
    // matrix P = transform::perspective(
    //     0.001f,
    //     10000.0f,
    //     aspect, clamped_fov
    // );

    // if (flip_x)
    // {
    //     P.a[0][0] = -P.a[0][0];
    // }
    // projection_matrix         = P;
    // inverse_projection_matrix = P.invert();
    //
    // const float w_lens          = 2.0f / P.a[1][1];
    // const float aspect_signed   = P.a[0][0] / P.a[1][1];
    // const float h_lens          = w_lens * aspect_signed;

    const float old_tan = 1.0f / std::fabs(projection_matrix.a[1][1]);
    const float new_tan = std::tan(clamped_fov * 0.5f * math::pi<float> / 180.0f);
    const float ratio   = old_tan / new_tan;

    projection_matrix.a[0][0] *= ratio;
    projection_matrix.a[1][1] *= ratio;
    inverse_projection_matrix = projection_matrix.invert();

    const float w_lens = 2.0f / projection_matrix.a[1][1];
    const float h_lens = w_lens * (projection_matrix.a[0][0] / projection_matrix.a[1][1]);
    a_film  = std::fabs(w_lens * h_lens);
    fov_deg = clamped_fov;

    a_film  = std::fabs(w_lens * h_lens);
    fov_deg = clamped_fov;
    world_to_clip = projection_matrix * camera_to_view;
    update_basis_cache();
}

fox_tracer::geometry::ray fox_tracer::camera::generate_ray(const float x, const float y) const
{
    const float xc = 2.0f * (x / width)  - 1.0f;
    const float yc = 1.0f - 2.0f * (y / height);

    // const vec3 dir_camera = inverse_projection_matrix
    //                           .mul_point_and_perspective_divide(vec3(xc, yc, 0.0f));
    // vec3 dir = camera_to_world.mul_vec(dir_camera);
    // dir = dir.normalize();

    const vec3 dir = (forward_ws + right_ws * xc + up_ws * yc).normalize();
    return {origin, dir};
}

fox_tracer::geometry::ray fox_tracer::camera::generate_ray_thin_lens(
    const float x, const float y, sampler *s) const
{
    if (lens_radius <= 0.0f)
    {
        return generate_ray(x, y);
    }

    // Pinhole direction in camera space
    const float xc = 2.0f * (x / width)  - 1.0f;
    const float yc = 1.0f - 2.0f * (y / height);

    const vec3 dir_camera = inverse_projection_matrix
                                    .mul_point_and_perspective_divide(
                                        vec3(xc, yc, 0.0f));

    // const vec3 x_focus_cam  = dir_camera * (focal_distance / dir_camera.z);
    // const vec3 lens_cam     = sampling::uniform_sample_disk(s->next(), s->next())
    //                             * lens_radius;
    //
    // const vec3 world_origin = camera_to_world.mul_point(lens_cam);
    // const vec3 focus_ws  = camera_to_world.mul_point(x_focus_cam);
    // const vec3 world_dir    = (focus_ws - world_origin).normalize();

    const vec3 lens_sample = sampling::uniform_sample_disk(s->next(), s->next());
    const vec3 x_lens      = lens_sample * lens_radius;

    float wz = dir_camera.z;
    if (std::fabs(wz) < 1e-8f) wz = (wz < 0.0f) ? -1e-8f : 1e-8f;
    const float    t_focus = focal_distance / wz;
    const vec3 x_focus = dir_camera * t_focus;

    const vec3 new_dir_camera = (x_focus - x_lens).normalize();

    const vec3 world_origin = camera_to_world.mul_point(x_lens);
    const vec3 world_dir    = camera_to_world.mul_vec(new_dir_camera).normalize();
    return {world_origin, world_dir};
}

bool fox_tracer::camera::project_onto_camera(const vec3 &p, float &x, float &y) const
{
    const vec3 pview = camera_to_view.mul_point(p);
    if (pview.z >= 0.0f) return false;

    //const vec3 pclip = projection_matrix.mul_point_and_perspective_divide(pview);
    const vec3 pclip = world_to_clip.mul_point_and_perspective_divide(p);

    // x = (pclip.x + 1.0f) * 0.5f;
    // y = (pclip.y + 1.0f) * 0.5f;
    //
    // if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f)
    // {
    //     return false;
    // }
    //
    // x = x * width;
    // y = 1.0f - y;
    // y = y * height;

    if (pclip.x < -1.0f || pclip.x > 1.0f ||
        pclip.y < -1.0f || pclip.y > 1.0f) return false;

    x = (pclip.x + 1.0f) * 0.5f * width;
    y = (1.0f - (pclip.y + 1.0f) * 0.5f) * height;
    return true;
}

void fox_tracer::camera::update_basis_cache() noexcept
{
    const vec3 p00 = inverse_projection_matrix
                        .mul_point_and_perspective_divide(vec3(0.0f, 0.0f, 0.0f));
    const vec3 p10 = inverse_projection_matrix
                            .mul_point_and_perspective_divide(vec3(1.0f, 0.0f, 0.0f));
    const vec3 p01 = inverse_projection_matrix
                            .mul_point_and_perspective_divide(vec3(0.0f, 1.0f, 0.0f));

    forward_ws = camera_to_world.mul_vec(p00);
    right_ws   = camera_to_world.mul_vec(p10 - p00);
    up_ws      = camera_to_world.mul_vec(p01 - p00);
}

fox_tracer::texture_cache::~texture_cache()
{
    for (auto& kv : textures_)
    {
        delete kv.second;
    }
}

fox_tracer::texture * fox_tracer::texture_cache::get_or_load(const std::string &filename)
{
    {
        std::shared_lock<std::shared_mutex> rlock(mtx_);
        const auto it = textures_.find(filename);
        if (it != textures_.end()) return it->second;
    }
    auto* tex = new texture();
    tex->load(filename);

    if (tex->texels == nullptr || tex->width <= 0 || tex->height <= 0)
    {
        LOG_WARN("texture") << "load failed (using default): " << filename;
    }
    else
    {
        LOG_DEBUG("texture") << "loaded " << filename
                             << " (" << tex->width << "x" << tex->height
                             << ", ch=" << tex->channels << ")";
    }

    std::unique_lock<std::shared_mutex> wlock(mtx_);
    const auto it = textures_.find(filename);
    if (it != textures_.end())
    {
        delete tex;
        return it->second;
    }
    textures_.insert({filename, tex});
    return tex;
}

fox_tracer::scene::container::~container()
{
    delete bvh;

    for (const lights::base* l : lights)
    {
        if (l != background)
        {
            delete l;
        }
    }
    delete background;
    for (const bsdf::base* m : materials)
    {
        delete m;
    }
}

void fox_tracer::scene::container::init(
    const std::vector<geometry::triangle> &mesh_triangles,
    const std::vector<bsdf::base *> &mesh_materials,
    lights::base *_background)
{
    for (const auto& t : mesh_triangles)
    {
        triangles.push_back(t);
        bounds.extend(t.vertices[0].position);
        bounds.extend(t.vertices[1].position);
        bounds.extend(t.vertices[2].position);
    }
    for (bsdf::base* m: mesh_materials)
    {
        materials.push_back(m);
    }
    background = _background;
    if (background != nullptr && background->total_integrated_power() > 0.0f)
    {
        background_light_idx = static_cast<int>(lights.size());
        lights.push_back(background);
    }
    else
    {
        background_light_idx = -1;
    }
}

void fox_tracer::scene::container::build()
{
    std::vector<geometry::triangle> input_triangles;
    input_triangles.reserve(triangles.size());
    for (const auto& t : triangles)
    {
        input_triangles.push_back(t);
    }
    triangles.clear();

    delete bvh;
    bvh = nullptr;

    if (!input_triangles.empty())
    {
        bvh = new accelerated_structure::bvh_node();
        bvh->build(input_triangles, triangles);
    }

    triangle_to_light.assign(triangles.size(), -1);
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        if (materials[triangles[i].material_index]->is_light())
        {
            auto* al = new lights::area();
            al->tri      = &triangles[i];
            al->emission = materials[triangles[i].material_index]->emission;
            triangle_to_light[i] = static_cast<int>(lights.size());
            lights.push_back(al);
        }
    }

    light_power_cdf.clear();
    light_power_cdf.reserve(lights.size());
    float running = 0.0f;
    for (lights::base* L : lights)
    {
        running += std::max(0.0f, L->total_integrated_power());
        light_power_cdf.push_back(running);
    }
}

fox_tracer::accelerated_structure::intersection_data fox_tracer::scene::container::traverse(
    const geometry::ray &r) const
{
    if (config().use_bvh.load(std::memory_order_relaxed) && bvh != nullptr)
    {
        return bvh->traverse(r, triangles);
    }
    accelerated_structure::intersection_data intersection;
    intersection.t = FLT_MAX;
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        float t, u, v;
        if (triangles[i].ray_intersect(r, t, u, v))
        {
            if (t < intersection.t)
            {
                intersection.t     = t;
                intersection.ID    = static_cast<unsigned int>(i);
                intersection.alpha = u;
                intersection.beta  = v;
                intersection.gamma = 1.0f - (u + v);
            }
        }
    }
    return intersection;
}

fox_tracer::lights::base * fox_tracer::scene::container::sample_light(
    sampler *s, float &pmf) const
{
    if (lights.empty())
    {
        pmf = 0.0f;
        return nullptr;
    }
    const int n = static_cast<int>(lights.size());

    const auto mode = static_cast<light_pick>(config().light_pick_mode.load(std::memory_order_relaxed));

    if (mode == light_pick::power_weighted
        && !light_power_cdf.empty()
        && light_power_cdf.back() > 0.0f)
    {
        const float total = light_power_cdf.back();
        const float u     = s->next() * total;

        auto it = std::ranges::lower_bound(light_power_cdf, u);

        int idx = static_cast<int>(it - light_power_cdf.begin());
        if (idx >= n) idx = n - 1;

        const float prev = (idx > 0) ? light_power_cdf[idx - 1] : 0.0f;
        pmf = (light_power_cdf[idx] - prev) / total;
        return lights[idx];
    }

    const int idx = std::min(static_cast<int>(s->next() * static_cast<float>(n)), n - 1);
    pmf = 1.0f / static_cast<float>(n);
    return lights[idx];
}

bool fox_tracer::scene::container::visible(const vec3 &p1, const vec3 &p2) const
{
    return visible(p1, vec3(0.0f, 0.0f, 0.0f), p2, vec3(0.0f, 0.0f, 0.0f));
}

bool fox_tracer::scene::container::visible(
    const vec3 &p1, const vec3 &g_n1,
    const vec3 &p2, const vec3 &g_n2) const
{
    vec3 dir = p2 - p1;
    const float dist = dir.length();
    if (dist <= 0.0f) return true;
    dir = dir / dist;

    const vec3 o1 = math::offset_ray_origin(p1, g_n1,  dir);
    const vec3 o2 = math::offset_ray_origin(p2, g_n2, -dir);

    geometry::ray r;
    r.init(o1, dir);

    const float max_t = (o2 - o1).length() - math::epsilon<float>;
    if (max_t <= 0.0f) return true;

    if (config().use_bvh.load(std::memory_order_relaxed) && bvh != nullptr)
    {
        return bvh->traverse_visible(r, triangles, max_t);
    }
    for (const auto& triangle : triangles)
    {
        float t, u, v;
        if (triangle.ray_intersect(r, t, u, v) && t < max_t)
        {
            return false;
        }
    }
    return true;
}

fox_tracer::color fox_tracer::scene::container::emit(
    const geometry::triangle *light_tri,
    const shading_data &sd,
    const vec3 &wi) const
{
    return materials[light_tri->material_index]->emit(sd, wi);
}

float fox_tracer::scene::container::light_pmf_by_index(int light_idx) const noexcept
{
    const int n = static_cast<int>(lights.size());
    if (n <= 0 || light_idx < 0 || light_idx >= n) return 0.0f;

    const auto mode = static_cast<light_pick>(
        config().light_pick_mode.load(std::memory_order_relaxed));

    if (mode == light_pick::power_weighted
        && !light_power_cdf.empty()
        && light_power_cdf.back() > 0.0f)
    {
        const float total = light_power_cdf.back();
        const float prev  = (light_idx > 0) ? light_power_cdf[light_idx - 1] : 0.0f;
        return (light_power_cdf[light_idx] - prev) / total;
    }

    return 1.0f / static_cast<float>(n);
}

fox_tracer::shading_data fox_tracer::scene::container::calculate_shading_data(
    const accelerated_structure::intersection_data &intersection,
    const geometry::ray &r) const
{
    shading_data sd;

    if (intersection.t < FLT_MAX)
    {
        sd.x         = r.at(intersection.t);
        sd.g_normal  = triangles[intersection.ID].g_normal();

        triangles[intersection.ID].interpolate_attributes(
            intersection.alpha, intersection.beta, intersection.gamma,
            sd.s_normal, sd.tu, sd.tv);

        sd.surface_bsdf = materials[triangles[intersection.ID].material_index];
        sd.wo           = -r.dir;

        if (sd.surface_bsdf->is_two_sided())
        {
            if (math::dot(sd.wo, sd.s_normal) < 0.0f)
            {
                sd.s_normal = -sd.s_normal;
            }
            if (math::dot(sd.wo, sd.g_normal) < 0.0f)
            {
                sd.g_normal = -sd.g_normal;
            }
        }
        sd.shading_frame.from_vector(sd.s_normal);
        sd.t = intersection.t;
    }
    else
    {
        sd.wo = -r.dir;
        sd.t  = intersection.t;
    }
    return sd;
}
