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
#include "render/photon_gatherer.h"

#include "config.h"
#include "framework/lights.h"
#include "framework/materials.h"
#include "render/photon_map.h"
#include "render/renderer.h"
#include "sampler/sampling.h"
#include "scene/scene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

// TODO: progressive photon mapping
// TODO: cone filter or Silverman kernel given If I have time
// TODO: BSDF-aware kernel

namespace fox_tracer::render
{
    void photon_gatherer::init(scene::container* _scene,
                               const photon_map* g,
                               const photon_map* c) noexcept
    {
        target_scene = _scene;
        global_map   = g;
        caustic_map  = c;
    }

    color photon_gatherer::density_estimate(const photon_map& map,
                                            const shading_data& sd,
                                            const int k, const float r_max) const
    {
        // std::vector<const photon*> found;
        // found.reserve(static_cast<std::size_t>(std::max(1, k)));

        thread_local std::vector<const photon*> found;
        found.clear();

        if (static_cast<int>(found.capacity()) < k)
            found.reserve(static_cast<std::size_t>(k));

        float r = 0.0f;
        map.knn(sd.x, k, r_max, found, r);
        if (found.empty() || r <= 0.0f) return color(0.0f, 0.0f, 0.0f);

        const float r2 = r * r;

        color flux(0.0f, 0.0f, 0.0f);

        const std::size_t       n    = found.size();
        const photon* const* const data = found.data();

        for (std::size_t i = 0; i < n; ++i)
        {
            const photon* p = data[i];
            flux = flux + sd.surface_bsdf->evaluate(sd, -p->wi) * p->power;
        }

        //~ test: cone filter
        // const float k_cone = 1.1f;
        // const float inv_kr = 1.0f / (k_cone * r);
        // for (std::size_t i = 0; i < n; ++i)
        // {
        //     const photon* p = data[i];
        //     const float dp = (p->position - sd.x).length();
        //     const float w  = std::max(0.0f, 1.0f - dp * inv_kr);
        //     flux = flux + sd.surface_bsdf->evaluate(sd, -p->wi) * p->power * w;
        // }
        // const float cone_area = (1.0f - 2.0f / (3.0f * k_cone)) * math::pi<float> * r2;
        // return flux * (1.0f / cone_area);

        //~ test: gaussian filter
        // const float alpha = 0.918f;
        // const float beta  = 1.953f;
        // const float inv_r2 = 1.0f / r2;
        // const float denom  = 1.0f - std::exp(-beta);
        // for (std::size_t i = 0; i < n; ++i)
        // {
        //     const photon* p = data[i];
        //     const float dp2 = (p->position - sd.x).length_squared();
        //     const float num = 1.0f - std::exp(-beta * 0.5f * dp2 * inv_r2);
        //     const float w   = alpha * (1.0f - num / denom);
        //     flux = flux + sd.surface_bsdf->evaluate(sd, -p->wi) * p->power * w;
        // }
        // return flux * (1.0f / (math::pi<float> * r2));

        // return flux / (math::pi<float> * r2);
        const float inv_area = 1.0f / (math::pi<float> * r2);
        return flux * inv_area;
    }

    color photon_gatherer::final_gather(const shading_data& sd, sampler* s,
                                        const direct_cb& compute_direct_cb) const
    {
        if (global_map == nullptr) return color(0.0f, 0.0f, 0.0f);
        if (final_gather_rays <= 0) return color(0.0f, 0.0f, 0.0f);

        color L(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < final_gather_rays; ++i)
        {
            color bsdf_w;
            float pdf = 0.0f;
            const vec3 wi = sd.surface_bsdf->sample(sd, s, bsdf_w, pdf);
            if (pdf <= 0.0f) continue;

            //~ kills bright fireflies but variance goes nuts
            // const float rr = std::min(0.95f, bsdf_w.luminance());
            // if (s->next() > rr) continue;
            // bsdf_w = bsdf_w * (1.0f / rr);

            geometry::ray r2;
            // r2.init(sd.x + wi * math::epsilon<float>, wi);
            r2.init(math::offset_ray_origin(sd.x, sd.g_normal, wi), wi);
            const accelerated_structure::intersection_data hit = target_scene->traverse(r2);
            if (hit.t >= FLT_MAX) continue;

            const shading_data sd2 = target_scene->calculate_shading_data(hit, r2);
            if (sd2.surface_bsdf == nullptr) continue;

            if (sd2.surface_bsdf->is_pure_specular()) continue;
            if (sd2.surface_bsdf->is_light()) continue;

            //~ TODO: Complete this later test out infinite loop (only if I have time its huge!)
            // if (sd2.surface_bsdf->is_pure_specular())
            // {
            //     color w2; float p2 = 0.0f;
            //     const vec3 wi2 = sd2.surface_bsdf->sample(sd2, s, w2, p2);
            //
            //     if (p2 <= 0.0f) continue;
            //
            //     geometry::ray r3;
            //     r3.init(sd2.x + wi2 * math::epsilon<float>, wi2);
            //
            //     const auto hit2 = target_scene->traverse(r3);
            // }

            const color indirect = density_estimate(*global_map, sd2,
                                                    k_global, r_max_global);

            color caustic_at_gather(0.0f, 0.0f, 0.0f);
            if (caustic_map != nullptr)
            {
                caustic_at_gather = density_estimate(*caustic_map, sd2,
                                                     k_caustic, r_max_caustic);
            }

            //~ test: without caustic at gather
            // L = L + bsdf_w * indirect * (cos_theta / pdf);

            const float cos_theta = std::fabs(math::dot(wi, sd.s_normal));
            L = L + bsdf_w * (indirect + caustic_at_gather) * (cos_theta / pdf);
        }

        // return L / static_cast<float>(final_gather_rays);
        const float inv_n = 1.0f / static_cast<float>(final_gather_rays);
        return L * inv_n;
    }

    color photon_gatherer::shade_eye(geometry::ray r, sampler* s,
                                     const direct_cb& compute_direct_cb) const
    {
        if (target_scene == nullptr) return color(0.0f, 0.0f, 0.0f);

        const int max_depth = std::max(1,
            config().max_depth.load(std::memory_order_relaxed));

        color throughput(1.0f, 1.0f, 1.0f);

        for (int depth = 0; depth < max_depth; ++depth)
        {
            const accelerated_structure::intersection_data
                     intersection = target_scene->traverse(r);
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
                return throughput * sd.surface_bsdf->emit(sd, sd.wo);

            if (!sd.surface_bsdf->is_pure_specular())
            {
                const color L_direct = compute_direct_cb(sd, s);

                color L_indirect(0.0f, 0.0f, 0.0f);
                if (global_map != nullptr)
                {
                    if (use_final_gather)
                    {
                        L_indirect = final_gather(sd, s, compute_direct_cb);
                    }
                    else
                    {
                        L_indirect = density_estimate(*global_map, sd,
                                                      k_global, r_max_global);
                    }
                }

                color L_caustic(0.0f, 0.0f, 0.0f);
                if (caustic_map != nullptr)
                {
                    L_caustic = density_estimate(*caustic_map, sd,
                                                 k_caustic, r_max_caustic);
                }

                return throughput * (L_direct + L_indirect + L_caustic);

                // const float w_pm = 0.5f;
                // const float w_fg = 0.5f;
                // return throughput * (L_direct + L_caustic
                //                    + w_pm * density_estimate(*global_map, sd, k_global, r_max_global)
                //                    + w_fg * final_gather(sd, s, compute_direct_cb));
            }

            color bsdf_weight;
            float pdf_bsdf = 0.0f;
            const vec3 wi_next = sd.surface_bsdf->sample(sd, s,
                                                             bsdf_weight, pdf_bsdf);
            if (pdf_bsdf <= 0.0f) return color(0.0f, 0.0f, 0.0f);

            const float cos_theta = std::fabs(math::dot(wi_next, sd.s_normal));
            throughput = throughput * bsdf_weight * (cos_theta / pdf_bsdf);
            if (throughput.luminance() <= 0.0f) return color(0.0f, 0.0f, 0.0f);

            // if (depth >= 3)
            // {
            //     const float rr = std::min(0.95f, throughput.luminance());
            //     if (rr <= 0.0f) return color(0.0f, 0.0f, 0.0f);
            //     if (s->next() > rr) return color(0.0f, 0.0f, 0.0f);
            //     throughput = throughput * (1.0f / rr);
            // }

            r.init(sd.x + wi_next * math::epsilon<float>, wi_next);
        }

        return color(0.0f, 0.0f, 0.0f);
    }
} // namespace fox_tracer
