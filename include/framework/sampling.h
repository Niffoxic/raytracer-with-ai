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
#ifndef RAYTRACER_WITH_AI_SAMPLING_H
#define RAYTRACER_WITH_AI_SAMPLING_H

#include "core.h"
#include <random>

namespace fox_tracer
{
    class sampler
    {
    public:
        virtual ~sampler  () = default;
        virtual float next() = 0;
    };

    class mt_random : public sampler
    {
    public:
        std::mt19937                          generator;
        std::uniform_real_distribution<float> dist;

        explicit mt_random(unsigned int seed = 1);
        float next() override;
    };

    namespace sampling
    {
        //~ Hemisphere uniform: random direction somewhere upper dome all are equally likely
        [[nodiscard]] vec3  uniform_sample_hemisphere(float r1, float r2) noexcept;
        [[nodiscard]] float uniform_hemisphere_pdf   (const vec3& wi)     noexcept;

        //~ Hemisphere cosine: biased toward the top factor out matte materials (diffuse best)
        [[nodiscard]] vec3  cosine_sample_hemisphere(float r1, float r2)  noexcept;
        [[nodiscard]] float cosine_hemisphere_pdf   (const vec3& wi)      noexcept;

        //~ Sphere uniform: full directional, to be used for my critical review testing
        // for smokes and fog (given if I find time) TODO: Implement FOG and Smoke and water maybe
        [[nodiscard]] vec3  uniform_sample_sphere(float r1, float r2) noexcept;
        [[nodiscard]] float uniform_sphere_pdf   (const vec3& wi)     noexcept;

        //~ returns position from randoms, TODO: depth of field
        [[nodiscard]] vec3 uniform_sample_disk(float r1, float r2) noexcept;

        // Isotropic: pick random direction inside a volume
        [[nodiscard]] vec3  sample_isotropic_phase(float r1, float r2) noexcept;
        [[nodiscard]] float isotropic_phase_pdf   ()                   noexcept;

        //  Henyey Greenstein: experimental picks a scattering direction (realistic anisotropic scattering)
        [[nodiscard]] vec3  sample_henyey_greenstein_local(float r1, float r2, float g) noexcept;
        [[nodiscard]] float henyey_greenstein_pdf         (float cos_theta, float g)    noexcept;
    }
}

#endif //RAYTRACER_WITH_AI_SAMPLING_H
