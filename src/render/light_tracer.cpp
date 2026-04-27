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
#include "render/light_tracer.h"

#include "config.h"
#include "framework/geometry.h"
#include "framework/imaging.h"
#include "framework/lights.h"
#include "framework/materials.h"
#include "sampler/sampling.h"
#include "scene/scene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

// TODO: thin-lens connection
// TODO: MIS weights between light tracer and path tracer

namespace fox_tracer::render
{
    void light_tracer::init(scene::container* _scene, film* _film,
                            std::mutex* _stripes, std::size_t _num_stripes,
                            std::size_t _paths_per_pass) noexcept
    {
        target_scene   = _scene;
        target_film    = _film;
        stripes        = _stripes;
        num_stripes    = _num_stripes;
        paths_per_pass = _paths_per_pass;
    }

    void light_tracer::splat_stripe(const float x, const float y, const color& L)
    {
        if (target_film->filter == nullptr) return;

        const int fw = static_cast<int>(target_film->width);
        const int fh = static_cast<int>(target_film->height);
        const int s  = target_film->filter->size();

        const int base_x = static_cast<int>(x);
        const int base_y = static_cast<int>(y);

        if (s == 0)
        {
            if (base_x < 0 || base_x >= fw || base_y < 0 || base_y >= fh) return;
            const std::size_t stripe = static_cast<unsigned>(base_y)
                                     & (num_stripes - 1);
            std::lock_guard<std::mutex> lk(stripes[stripe]);
            target_film->film_buffer[base_y * fw + base_x] =
                target_film->film_buffer[base_y * fw + base_x] + L;
            return;
        }

        // for (int i = -s; i <= s; ++i)
        // for (int j = -s; j <= s; ++j)
        // {
        //     const int py = base_y + i;
        //     const int px = base_x + j;
        //     if (px < 0 || px >= fw || py < 0 || py >= fh) continue;
        //
        //     const float w = target_film->filter->filter(static_cast<float>(px) - x, static_cast<float>(py) - y);
        //
        //     if (w == 0.0f) continue;
        //
        //     const std::size_t stripe = static_cast<unsigned>(py) & (num_stripes - 1);
        //     std::lock_guard<std::mutex> lk(stripes[stripe]);
        //     target_film->film_buffer[py * fw + px] = target_film->film_buffer[py * fw + px] + L * w;
        // }

        float weights[25];
        int   px_row[25];
        int   rows_py[25];
        int   row_begin[25];
        int   row_count[25];
        int   n_rows = 0;
        int   n_taps = 0;
        float total  = 0.0f;

        for (int i = -s; i <= s; ++i)
        {
            const int py = base_y + i;
            if (py < 0 || py >= fh) continue;
            const float dy = static_cast<float>(py) - y;

            const int row_start = n_taps;
            for (int j = -s; j <= s; ++j)
            {
                const int px = base_x + j;
                if (px < 0 || px >= fw) continue;

                const float w = target_film->filter->filter(
                    static_cast<float>(px) - x, dy);
                if (w == 0.0f) continue;

                weights[n_taps] = w;
                px_row [n_taps] = px;
                total          += w;
                ++n_taps;
            }
            const int row_len = n_taps - row_start;
            if (row_len > 0)
            {
                rows_py  [n_rows] = py;
                row_begin[n_rows] = row_start;
                row_count[n_rows] = row_len;
                ++n_rows;
            }
        }
        if (total <= 0.0f) return;
        const float inv_total = 1.0f / total;

        for (int r = 0; r < n_rows; ++r)
        {
            const int py = rows_py[r];
            const std::size_t stripe = static_cast<unsigned>(py)
                                     & (num_stripes - 1);
            std::lock_guard<std::mutex> lk(stripes[stripe]);
            const int i0 = row_begin[r];
            const int i1 = i0 + row_count[r];
            for (int i = i0; i < i1; ++i)
            {
                const int idx = py * fw + px_row[i];
                target_film->film_buffer[idx] =
                    target_film->film_buffer[idx]
                    + (L * (weights[i] * inv_total));
            }
        }
    }

    void light_tracer::connect_to_camera(const vec3& p, const vec3& n,
                                         const color& col)
    {
        if (target_scene == nullptr || target_film == nullptr) return;
        const camera& cam = target_scene->cam;

        const vec3  to_cam = cam.origin - p;
        const float d2     = to_cam.length_squared();
        if (d2 <= math::squared(math::epsilon<float>)) return;

        // const float d  = std::sqrt(d2)
        // const vec3  wi = to_cam / d;
        const float inv_d = 1.0f / std::sqrt(d2);
        const float d     = d2 * inv_d;
        const vec3  wi    = to_cam * inv_d;

        const vec3  dir_from_cam = -wi;
        const float cos_cam      = math::dot(dir_from_cam, cam.view_direction);
        if (cos_cam <= 0.0f) return;

        const float cos_surface = math::dot(wi, n);
        if (cos_surface <= 0.0f) return;

        // const float cos_s = math::dot(wi, sd.s_normal);
        // const float cos_g = math::dot(wi, sd.g_normal);
        // const float cos_surface = std::min(cos_s, cos_g);
        // if (cos_surface <= 0.0f) return;

        float img_x, img_y;
        if (!cam.project_onto_camera(p, img_x, img_y)) return;

        //if (!target_scene->visible(p, cam.origin)) return;
        if (!target_scene->visible(p, n, cam.origin, vec3(0.0f, 0.0f, 0.0f)))
            return;

        const float G = cos_surface * cos_cam / d2;
        // const float G = cos_surface * cos_cam;
        // const float G = cos_surface / d2;

        const float cos_cam2 = cos_cam * cos_cam;
        const float cos_cam4 = cos_cam2 * cos_cam2;
        const float W_e = 1.0f / (cam.a_film * cos_cam4);

        // const float W_e = 1.0f / (cam.a_film * cos_cam2 * cos_cam);
        // const float W_e = 1.0f / cam.a_film;

        const float inv_n = (paths_per_pass > 0)
                          ? (1.0f / static_cast<float>(paths_per_pass))
                          : 1.0f;

        splat_stripe(img_x, img_y, col * (G * W_e * inv_n));
    }

    void light_tracer::light_trace(sampler* s)
    {
        if (target_scene == nullptr) return;
        if (target_scene->lights.empty()) return;

        float pmf = 0.0f;
        lights::base* L_src = target_scene->sample_light(s, pmf);
        if (L_src == nullptr || pmf <= 0.0f) return;

        float pdf_pos = 0.0f;
        const vec3 p = L_src->sample_position_from_light(s, pdf_pos);
        if (pdf_pos <= 0.0f) return;

        float pdf_dir = 0.0f;
        const vec3 wi = L_src->sample_direction_from_light(s, pdf_dir);
        if (pdf_dir <= 0.0f) return;

        const color Le = L_src->evaluate(-wi);
        if (Le.luminance() <= 0.0f) return;

        const vec3 n_light = L_src->normal(shading_data(p, wi), wi);

        const float inv_pos_pmf = 1.0f / (pdf_pos * pmf);
        connect_to_camera(p, n_light, Le * inv_pos_pmf);

        //~ W_light = (cos_theta_wi / (pdf_pos * pdf_dir)) / pmf
        const float cos_theta_wi = L_src->is_area()
                                 ? std::max(0.0f, math::dot(wi, n_light))
                                 : 1.0f;
        if (cos_theta_wi <= 0.0f) return;

        const float inv_emit_pdf = 1.0f / (pdf_pos * pdf_dir * pmf);
        const color throughput(cos_theta_wi * inv_emit_pdf,
                               cos_theta_wi * inv_emit_pdf,
                               cos_theta_wi * inv_emit_pdf);

        geometry::ray r;
        r.init(math::offset_ray_origin(p, n_light, wi), wi);
        light_trace_path(r, throughput, Le, s);
    }

    void light_tracer::light_trace_path(geometry::ray& r,
                                        const color& path_throughput,
                                        const color& Le,
                                        sampler* s)
    {
        if (target_scene == nullptr) return;

        const int  max_depth = std::max(1, config().max_depth.load(std::memory_order_relaxed));
        const bool use_rr    = config().use_rr.load(std::memory_order_relaxed);
        const int  rr_depth  = std::max(0, config().rr_depth.load(std::memory_order_relaxed));

        const vec3 cam_origin = target_scene->cam.origin;

        color throughput = path_throughput;

        for (int depth = 0; depth < max_depth; ++depth)
        {
            const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
            const shading_data sd = target_scene->calculate_shading_data(intersection, r);

            if (sd.t >= FLT_MAX) return;
            if (sd.surface_bsdf == nullptr) return;

            if (sd.surface_bsdf->is_light()) return;

            if (!sd.surface_bsdf->is_pure_specular())
            {
                // const vec3 wi_cam = (cam_origin - sd.x).normalize();
                const vec3 wi_cam = (cam_origin - sd.x).normalize();
                const color f_s   = sd.surface_bsdf->evaluate(sd, wi_cam);
                if (f_s.luminance() > 0.0f)
                {
                    connect_to_camera(sd.x, sd.s_normal,
                                      throughput * f_s * Le);
                }
            }

            if (use_rr && depth >= rr_depth)
            {
                const float rr = std::min(0.95f,
                            std::max(throughput.red,
                            std::max(throughput.green, throughput.blue)));

                // const float rr = std::min(0.95f, throughput.luminance());
                // const float rr = 0.5f;

                if (rr <= 0.0f) return;
                if (s->next() > rr) return;

                // throughput = throughput / rr;
                throughput = throughput * (1.0f / rr);
            }

            color bsdf_weight;
            float pdf_bsdf = 0.0f;
            const vec3 wi_next = sd.surface_bsdf->sample(sd, s,
                                                             bsdf_weight, pdf_bsdf);
            if (pdf_bsdf <= 0.0f) return;
            const float corr = 1.0f;

            // const vec3 wo_here = sd.wo; //~ prev ray reversed
            // const float num = std::fabs(math::dot(wi_next, sd.s_normal))
            //                 * std::fabs(math::dot(wo_here,  sd.g_normal));
            // const float den = std::max(1e-6f,
            //                     std::fabs(math::dot(wi_next, sd.g_normal))
            //                   * std::fabs(math::dot(wo_here,  sd.s_normal)));
            // const float corr = num / den;

            const float cos_theta = std::fabs(math::dot(wi_next, sd.s_normal));
            throughput = throughput * bsdf_weight * (cos_theta * corr / pdf_bsdf);
            if (throughput.luminance() <= 0.0f) return;

            r.init(math::offset_ray_origin(sd.x, sd.g_normal, wi_next), wi_next);
        }
    }
} // namespace fox_tracer
