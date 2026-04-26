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

#include "sampler/sampling.h"
#include "config.h"

#include <algorithm>
#include <cmath>
#include <cfloat>

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

    const vec3 pclip = projection_matrix.mul_point_and_perspective_divide(pview);

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