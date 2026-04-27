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
#ifndef RAYTRACER_WITH_AI_PHOTON_GATHERER_H
#define RAYTRACER_WITH_AI_PHOTON_GATHERER_H

#include "framework/core.h"
#include "framework/geometry.h"
#include "framework/materials.h"

#include <functional>

namespace fox_tracer
{
    namespace scene
    {
        class container;
    }
    class sampler;

}

namespace fox_tracer::render
{
    class photon_map;
    class ray_tracer;

    class photon_gatherer
    {
    public:
        using direct_cb = std::function<color(const shading_data&, sampler*)>;

        scene::container* target_scene{ nullptr };
        const photon_map* global_map  { nullptr };
        const photon_map* caustic_map { nullptr };

        ray_tracer* owner{nullptr};

        int   k_global       {100};
        int   k_caustic      {60};
        float r_max_global   {0.5f};
        float r_max_caustic  {0.1f};

        bool  use_final_gather{false};
        int   final_gather_rays{16};

        void init(scene::container* _scene,
                  const photon_map* g,
                  const photon_map* c) noexcept;

        [[nodiscard]] color shade_eye(geometry::ray r, sampler* s,
                                      const direct_cb& compute_direct_cb) const;

    private:
        [[nodiscard]] color density_estimate(const photon_map& map,
                                             const shading_data& sd,
                                             int k, float r_max) const;

        [[nodiscard]] color final_gather(const shading_data& sd, sampler* s,
                                         const direct_cb& compute_direct_cb) const;
    };
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_PHOTON_GATHERER_H
