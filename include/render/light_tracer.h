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
#ifndef RAYTRACER_WITH_AI_LIGHT_TRACER_H
#define RAYTRACER_WITH_AI_LIGHT_TRACER_H

#include "framework/core.h"
#include "framework/geometry.h"

#include <cstddef>
#include <mutex>

namespace fox_tracer
{
    namespace scene
    {
        class container;
    }
    class film;
    class sampler;
}

namespace fox_tracer::render
{
    //~ Unbiased Non-uniform noise
    class light_tracer
    {
    public:
        scene::container* target_scene{nullptr};
        film*             target_film {nullptr};
        std::mutex*       stripes     {nullptr};
        std::size_t       num_stripes {0};

        std::size_t paths_per_pass{0};

        void init(scene::container* _scene, film* _film,
                  std::mutex* _stripes, std::size_t _num_stripes,
                  std::size_t _paths_per_pass) noexcept;

        void connect_to_camera(const vec3& p, const vec3& n, const color& col);

        void light_trace     (sampler* s);
        void light_trace_path(geometry::ray& r,
                              const color& path_throughput,
                              const color& Le,
                              sampler* s);

    private:
        void splat_stripe(float x, float y, const color& L);
    };
} // namespace fox_tracer::render

#endif // RAYTRACER_WITH_AI_LIGHT_TRACER_H
