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
#ifndef RAYTRACER_WITH_AI_PHOTON_SHOOTER_H
#define RAYTRACER_WITH_AI_PHOTON_SHOOTER_H

#include "framework/core.h"
#include "framework/geometry.h"

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

    class photon_shooter
    {
    public:
        scene::container*   target_scene{nullptr};
        photon_map*         global_map  {nullptr};
        photon_map*         caustic_map {nullptr};

        // ~ quick debug
        // std::size_t p_global  {20'000};
        // std::size_t p_caustic {5'000};

        std::size_t p_global  {200'000};
        std::size_t p_caustic {50'000};

        void init(scene::container* _scene, photon_map* g, photon_map* c) noexcept;

        void shoot_one(sampler* s, bool for_caustic_map, int worker_id);

    private:
        void trace(geometry::ray&   r,
                   color            power,
                   bool             for_caustic_map,
                   bool             saw_specular_so_far,
                   sampler*         s,
                   int              worker_id);
    };
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_PHOTON_SHOOTER_H
