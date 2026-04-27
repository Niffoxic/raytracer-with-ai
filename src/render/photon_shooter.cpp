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
#include "render/photon_shooter.h"
#include "render/photon_map.h"

#include "config.h"
#include "framework/lights.h"
#include "framework/materials.h"
#include "sampler/sampling.h"
#include "scene/scene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

// TODO: importance-driven photon emission
// TODO: photon RR based on path length not just throughput
// TODO: try storing photons at specular hits with delta BSDF flag for SDS paths

namespace fox_tracer::render
{
    void photon_shooter::init(scene::container* _scene, photon_map* g, photon_map* c) noexcept
    {
        target_scene = _scene;
        global_map   = g;
        caustic_map  = c;
    }

    void photon_shooter::shoot_one(
        sampler*    s,
        const bool  for_caustic_map,
        const int   worker_id)
    {
        if (target_scene == nullptr) return;
        if (target_scene->lights.empty()) return;

        photon_map* dst = for_caustic_map ? caustic_map : global_map;
        if (dst == nullptr) return;

        //~ Light pick
        float pmf = 0.0f;
        lights::base* L_src = target_scene->sample_light(s, pmf);
        if (L_src == nullptr || pmf <= 0.0f) return;

        //~ Position on the light
        float pdf_pos = 0.0f;
        const vec3 p = L_src->sample_position_from_light(s, pdf_pos);
        if (pdf_pos <= 0.0f) return;

        //~ direction from the light
        float pdf_dir = 0.0f;
        const vec3 wi = L_src->sample_direction_from_light(s, pdf_dir);
        if (pdf_dir <= 0.0f) return;

        const color Le = L_src->evaluate(-wi);
        if (Le.luminance() <= 0.0f) return;

        const vec3 n_light = L_src->normal(shading_data(p, wi), wi);

        //~ Emitted throughput
        const std::size_t P = for_caustic_map ? p_caustic : p_global;
        if (P == 0) return;

        const float cos_theta_wi = L_src->is_area()
                                 ? std::max(0.0f, math::dot(wi, n_light))
                                 : 1.0f;
        if (cos_theta_wi <= 0.0f) return;

        // const color power = Le * cos_theta_wi / (pdf_pos * pdf_dir * pmf * static_cast<float>(P));

        const float inv_pdf = 1.0f / (pdf_pos * pdf_dir * pmf
                                     * static_cast<float>(P));
        const color power = Le * (cos_theta_wi * inv_pdf);

        // walk the scene
        geometry::ray r;
        // r.init(p + wi * math::epsilon<float>, wi);
        r.init(math::offset_ray_origin(p, n_light, wi), wi);
        trace(r, power,
            for_caustic_map,
            false, s,
            worker_id);
    }

    void photon_shooter::trace(geometry::ray& r,
                               color          power,
                               bool           for_caustic_map,
                               bool           saw_specular_so_far,
                               sampler*       s,
                               int            worker_id)
    {
        photon_map* dst = for_caustic_map ? caustic_map : global_map;
        if (dst == nullptr) return;

        const int  max_depth = std::max(1, config().max_depth.load(std::memory_order_relaxed));
        const bool use_rr    = config().use_rr.load(std::memory_order_relaxed);
        const int  rr_depth  = std::max(0, config().rr_depth.load(std::memory_order_relaxed));

        for (int depth = 0; depth < max_depth; ++depth)
        {
            const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
            const shading_data sd = target_scene->calculate_shading_data(intersection, r);

            if (sd.t >= FLT_MAX) return;
            if (sd.surface_bsdf == nullptr) return;
            if (sd.surface_bsdf->is_light()) return;

            const bool specular_here = sd.surface_bsdf->is_pure_specular();

            if (!specular_here)
            {
                //~ caustic map = LSDE only
                if (for_caustic_map)
                {
                    if (saw_specular_so_far)
                    {
                        photon ph;
                        ph.position = sd.x;
                        ph.wi       = -sd.wo;
                        ph.power    = power;
                        ph.s_normal = sd.s_normal;
                        dst->add(ph, worker_id);
                    }
                    //~ caustic walk dies at the first diffuse anyway
                    return;
                }

                //~ global map stores everywhere diffuse keeps walking keep walking
                photon ph;
                ph.position = sd.x;
                ph.wi       = -sd.wo;
                ph.power    = power;
                ph.s_normal = sd.s_normal;
                dst->add(ph, worker_id);
            }

            if (use_rr && depth >= rr_depth)
            {
                // const float rr = std::min(0.95f, power.luminance());
                const float rr = std::min(0.95f,
                            std::max(power.red,
                            std::max(power.green,power.blue)));
                if (rr <= 0.0f) return;
                if (s->next() > rr) return;

                // power = power / rr; //~ 3 divs
                power = power * (1.0f / rr);
            }

            color bsdf_weight;
            float pdf_bsdf = 0.0f;
            const vec3 wi_next = sd.surface_bsdf->sample(sd, s, bsdf_weight, pdf_bsdf);
            if (pdf_bsdf <= 0.0f) return;

            const float cos_theta = std::fabs(math::dot(wi_next, sd.s_normal));
            power = power * bsdf_weight * (cos_theta / pdf_bsdf);
            if (power.luminance() <= 0.0f) return;

            //~ MUST be set after the diffuse-store (i gotta do something about this one later)
            if (specular_here) saw_specular_so_far = true;

            // r.init(sd.x + wi_next * math::epsilon<float>, wi_next);
            r.init(math::offset_ray_origin(sd.x, sd.g_normal, wi_next), wi_next);
        }
    }
} // namespace fox_tracer
