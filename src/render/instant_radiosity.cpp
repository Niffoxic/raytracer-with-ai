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
#include "render/instant_radiosity.h"

#include "config.h"
#include "framework/geometry.h"
#include "framework/imaging.h"
#include "framework/lights.h"
#include "framework/materials.h"
#include "render/renderer.h"
#include "sampler/sampling.h"
#include "scene/scene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <immintrin.h>

// TODO: Bias-correction clamp for near-singular VPLs (maybe I can also try Bidirectional Instant Radiosity given if i have time
// TODO: Lock-free SPSC queues for pending_vpls

namespace fox_tracer::render
{
    void instant_radiosity::init(scene::container* _scene, film* _film,
                                 std::mutex* _stripes, std::size_t _num_stripes,
                                 std::size_t _num_vpls) noexcept
    {
        target_scene = _scene;
        target_film  = _film;
        stripes      = _stripes;
        num_stripes  = _num_stripes;
        num_vpls     = _num_vpls;

        // TODO: I gotta do something about reserving vlps
    }

    void instant_radiosity::resize_workers(int num_workers) noexcept
    {
        if (num_workers < 1) num_workers = 1;

        // const std::size_t per_worker = std::max<std::size_t>(64,
        //     num_vpls / static_cast<std::size_t>(num_workers));
        // for (auto& v : pending_vpls) v.reserve(per_worker);

        pending_vpls.assign(static_cast<std::size_t>(num_workers),
                            std::vector<vpl>{});

        for (auto& v : pending_vpls) v.reserve(64);
    }

    void instant_radiosity::merge_pending()
    {
        std::size_t total = 0;
        for (const auto& v : pending_vpls)
            total += v.size();

        if (total == 0) return;

        // for (auto& v : pending_vpls) //~ naah this is hella slow
        // {
        //     vpls.insert(vpls.end(), v.begin(), v.end());
        //     v.clear();
        // }

        // vpls.reserve(vpls.size() + total);
        // for (auto& v : pending_vpls) //~ still
        // {
        //     vpls.insert(vpls.end(),
        //                 std::make_move_iterator(v.begin()),
        //                 std::make_move_iterator(v.end()));
        //     v.clear();
        // }

        // if (vpls.empty() && !pending_vpls.empty()) //~ I can do better
        // {
        //     vpls = std::move(pending_vpls[0]);
        //     for (std::size_t i = 1; i < pending_vpls.size(); ++i)
        //     {
        //         vpls.insert(vpls.end(),
        //                     pending_vpls[i].begin(),
        //                     pending_vpls[i].end());
        //         pending_vpls[i].clear();
        //     }
        //     return;
        // }

        //~ bruh could have did it from the beginning
        vpls.reserve(vpls.size() + total);
        for (auto& v : pending_vpls)
        {
            vpls.insert(vpls.end(), v.begin(), v.end());
            v.clear();
        }
    }

    void instant_radiosity::shoot_one_path(sampler* s, int worker_id)
    {
        if (target_scene == nullptr) return;
        if (target_scene->lights.empty()) return;

        float pmf = 0.0f;
        lights::base* L_src = target_scene->sample_light(s, pmf);
        if (L_src == nullptr || pmf <= 0.0f) return;

        float pdf_pos = 0.0f;
        const vec3 p = L_src->sample_position_from_light(s, pdf_pos);
        if (pdf_pos <= 0.0f) return;

        const vec3 n_light = L_src->normal(shading_data(p, vec3()), vec3());
        const color Le_emit = L_src->evaluate(-n_light);
        if (Le_emit.luminance() > 0.0f)
        {
            vpl v;
            v.sd.x        = p;
            v.sd.s_normal = n_light;
            v.sd.g_normal = n_light;
            v.sd.wo       = n_light;
            v.sd.surface_bsdf = nullptr;
            // v.Le = Le_emit / (pdf_pos * pmf);
            const float inv_emit_pmf = 1.0f / (pdf_pos * pmf);
            v.Le = Le_emit * inv_emit_pmf;

            if (worker_id >= 0
                && static_cast<std::size_t>(worker_id) < pending_vpls.size())
            {
                pending_vpls[static_cast<std::size_t>(worker_id)]
                    .push_back(v);
            }
            else
            {
                std::lock_guard<std::mutex> lk(vpls_mtx);
                vpls.push_back(v);
            }
        }

        float pdf_dir = 0.0f;
        const vec3 wi = L_src->sample_direction_from_light(s, pdf_dir);
        if (pdf_dir <= 0.0f) return;

        const color Le = L_src->evaluate(-wi);
        if (Le.luminance() <= 0.0f) return;

        const float cos_theta_wi = L_src->is_area()
                                 ? std::max(0.0f, math::dot(wi, n_light))
                                 : 1.0f;
        if (cos_theta_wi <= 0.0f) return;

        // const color throughput = Le * cos_theta_wi / (pdf_pos * pdf_dir * pmf);

        const float inv_emit_pdf = 1.0f / (pdf_pos * pdf_dir * pmf);
        const color throughput = Le * (cos_theta_wi * inv_emit_pdf);

        geometry::ray r;
        r.init(p + wi * math::epsilon<float>, wi);
        trace_light_subpath(r, throughput, s, worker_id);
    }

    void instant_radiosity::trace_light_subpath(geometry::ray& r,
                                                const color& throughput_in,
                                                sampler* s,
                                                int worker_id)
    {
        const int  max_depth = std::max(1, config().max_depth.load(std::memory_order_relaxed));
        const bool use_rr    = config().use_rr.load(std::memory_order_relaxed);
        const int  rr_depth  = std::max(0, config().rr_depth.load(std::memory_order_relaxed));

        color throughput = throughput_in;

        std::vector<vpl>* my_bin = nullptr;
        if (worker_id >= 0
            && static_cast<std::size_t>(worker_id) < pending_vpls.size())
        {
            my_bin = &pending_vpls[static_cast<std::size_t>(worker_id)];
        }

        for (int depth = 0; depth < max_depth; ++depth)
        {
            const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
            const shading_data sd = target_scene->calculate_shading_data(intersection, r);

            if (sd.t >= FLT_MAX) return;
            if (sd.surface_bsdf == nullptr) return;
            if (sd.surface_bsdf->is_light()) return;

            if (!sd.surface_bsdf->is_pure_specular())
            {
                vpl v;
                v.sd = sd;
                v.Le = throughput;

                // if (worker_id >= 0
                //     && static_cast<std::size_t>(worker_id) < pending_vpls.size())
                // {
                //     pending_vpls[static_cast<std::size_t>(worker_id)]
                //         .push_back(v);
                // }
                // else
                // {
                //     std::lock_guard<std::mutex> lk(vpls_mtx);
                //     vpls.push_back(v);
                // }

                if (my_bin != nullptr)
                {
                    my_bin->push_back(v);
                }
                else
                {
                    std::lock_guard<std::mutex> lk(vpls_mtx);
                    vpls.push_back(v);
                }
            }

            if (use_rr && depth >= rr_depth)
            {
                // const float rr = std::min(0.95f, throughput.luminance());
                const float rr = std::min(0.95f,
                                    std::max(throughput.red,
                                    std::max(throughput.green,
                                    throughput.blue)));

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

            const float cos_theta = std::fabs(math::dot(wi_next, sd.s_normal));
            throughput = throughput * bsdf_weight * (cos_theta / pdf_bsdf);
            if (throughput.luminance() <= 0.0f) return;

            r.init(sd.x + wi_next * math::epsilon<float>, wi_next);
        }
    }

    color instant_radiosity::gather_indirect(const shading_data& sd,
                                             sampler* s) const
    {
        if (target_scene == nullptr)
            return color(0.0f, 0.0f, 0.0f);

        if (sd.surface_bsdf == nullptr)
            return color(0.0f, 0.0f, 0.0f);

        if (sd.surface_bsdf->is_pure_specular())
            return color(0.0f, 0.0f, 0.0f);

        if (vpls.empty())
            return color(0.0f, 0.0f, 0.0f);

        const float min_dist_sq = std::max(1e-8f,
            config().ir_min_dist_sq.load(std::memory_order_relaxed));

        color L(0.0f, 0.0f, 0.0f);

        const std::size_t n    = vpls.size();
        const vpl* const  data = vpls.data();

        for (std::size_t i = 0; i < n; ++i)
        {
            const vpl& y = data[i];

            vec3 w = y.sd.x - sd.x;
            const float d2 = w.length_squared();
            if (d2 <= math::squared(math::epsilon<float>)) continue;

            // const float d = std::sqrt(d2);
            // w = w / d;
            // const float inv_d = 1.0f / std::sqrt(d2);
            const float inv_d = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(d2)));
            w = w * inv_d;

            const float cos_x = std::max(0.0f,
                math::dot(w, sd.s_normal));
            const float cos_y = std::max(0.0f,
                math::dot(-w, y.sd.s_normal));
            if (cos_x <= 0.0f || cos_y <= 0.0f) continue;


            if (!target_scene->visible(sd.x, y.sd.x)) continue;
            // if (cos_x <= 0.0f || cos_y <= 0.0f) continue;

            const float r2c = std::max(d2, min_dist_sq);
            const float G   = (cos_x * cos_y) / r2c;

            const color f_x = sd.surface_bsdf->evaluate(sd, w);
            const color f_y = (y.sd.surface_bsdf == nullptr)
                              ? color(1.0f, 1.0f, 1.0f)
                              : y.sd.surface_bsdf->evaluate(y.sd, -w);

            L = L + f_x * f_y * y.Le * G;
        }

        const float inv_n = 1.0f / static_cast<float>(n);
        return L * inv_n;
    }

    color instant_radiosity::shade_eye(geometry::ray r, sampler* s) const
    {
        if (target_scene == nullptr) return color(0.0f, 0.0f, 0.0f);

        const int max_depth = std::max(1, config().max_depth.load(std::memory_order_relaxed));

        const float firefly_max_direct_L   = std::max(0.0f,
            config().pt_firefly_max_direct  .load(std::memory_order_relaxed));
        const float firefly_max_indirect_L = std::max(0.0f,
            config().pt_firefly_max_indirect.load(std::memory_order_relaxed));

        auto firefly = [](color c, float max_L) -> color
        {
            if (!std::isfinite(c.red) || !std::isfinite(c.green) || !std::isfinite(c.blue))
                return color(0.0f, 0.0f, 0.0f);
            if (max_L <= 0.0f) return c;
            //     return color(std::min(c.red,   max_L),
            //                  std::min(c.green, max_L),
            //                  std::min(c.blue,  max_L));
            const float lum = c.luminance();
            if (lum > max_L) c = c * (max_L / lum);
            return c;
        };

        color throughput(1.0f, 1.0f, 1.0f);

        for (int depth = 0; depth < max_depth; ++depth)
        {
            const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
            const shading_data sd = target_scene->calculate_shading_data(intersection, r);

            if (sd.t >= FLT_MAX)
            {
                if (owner != nullptr)
                {
                    return throughput * owner->evaluate_background(r);
                }
                if (config().override_background.load(std::memory_order_relaxed))
                {
                    const color bg(
                        config().bg_r.load(std::memory_order_relaxed),
                        config().bg_g.load(std::memory_order_relaxed),
                        config().bg_b.load(std::memory_order_relaxed));
                    return throughput * bg;
                }
                if (target_scene->background != nullptr)
                {
                    return throughput * target_scene->background->evaluate(r.dir);
                }
                return color(0.0f, 0.0f, 0.0f);
            }

            if (sd.surface_bsdf == nullptr)
                return color(0.0f, 0.0f, 0.0f);

            if (sd.surface_bsdf->is_light())
            {
                return throughput * sd.surface_bsdf->emit(sd, sd.wo);
            }

            if (!sd.surface_bsdf->is_pure_specular())
            {
                color L_direct(0.0f, 0.0f, 0.0f);
                if (owner != nullptr)
                {
                    L_direct = owner->compute_direct(sd, s);
                }
                color L_indirect = gather_indirect(sd, s);
                L_direct   = firefly(L_direct,   firefly_max_direct_L);
                L_indirect = firefly(L_indirect, firefly_max_indirect_L);
                return throughput * (L_direct + L_indirect);
            }

            color bsdf_weight;
            float pdf_bsdf = 0.0f;
            const vec3 wi_next = sd.surface_bsdf->sample(sd, s,
                                                             bsdf_weight, pdf_bsdf);
            if (pdf_bsdf <= 0.0f) return color(0.0f, 0.0f, 0.0f);

            const float cos_theta = std::fabs(math::dot(wi_next, sd.s_normal));
            throughput = throughput * bsdf_weight * (cos_theta / pdf_bsdf);
            if (throughput.luminance() <= 0.0f) return color(0.0f, 0.0f, 0.0f);

            r.init(sd.x + wi_next * math::epsilon<float>, wi_next);
        }

        return color(0.0f, 0.0f, 0.0f);
    }
} // namespace fox_tracer
