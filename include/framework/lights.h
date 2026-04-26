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
#ifndef RAYTRACER_WITH_AI_LIGHTS_H
#define RAYTRACER_WITH_AI_LIGHTS_H

#include "core.h"

#pragma warning( disable : 4244)

namespace fox_tracer
{
    namespace geometry
    {
        class triangle;
        class ray;
    }

    class texture;
    class sampler;
    class shading_data;

    class scene_bounds
    {
    public:
        vec3  scene_centre;
        float scene_radius{};
    };

    namespace lights
    {
        class base
        {
        public:
            virtual ~base() = default;

            virtual vec3  sample  ( const shading_data& sd, sampler* s,
                                    color& emitted_colour, float& pdf)      = 0;
            virtual color evaluate( const vec3& wi)                         = 0;
            virtual float pdf     ( const shading_data& sd, const vec3& wi) = 0;

            [[nodiscard]]
            virtual bool     is_area                () const = 0;
            virtual float    total_integrated_power ()       = 0;

            virtual vec3 normal  (const shading_data& sd, const vec3& wi) = 0;

            virtual vec3 sample_position_from_light (sampler* s, float& pdf) = 0;
            virtual vec3 sample_direction_from_light(sampler* s, float& pdf) = 0;
        };

        class area : public base
        {
            using triangle = geometry::triangle;
        public:
            triangle* tri{ nullptr };
            color     emission;

            vec3 sample     (const shading_data& sd, sampler* s,
                             color& emitted_colour, float& pdf)      override;
            color evaluate  (const vec3& wi)                         override;
            float pdf       (const shading_data& sd, const vec3& wi) override;

            [[nodiscard]]
            bool  is_area               () const override;
            float total_integrated_power() override;

            vec3 normal(const shading_data& sd, const vec3& wi) override;

            vec3 sample_position_from_light (sampler* s, float& pdf) override;
            vec3 sample_direction_from_light(sampler* s, float& pdf) override;
        };

        class background_colour : public base
        {
        public:
            color emission;

            explicit background_colour(const color& _emission) noexcept;

            vec3 sample     (const shading_data& sd, sampler* s,
                             color& emitted_colour, float& pdf)      override;
            color evaluate  (const vec3& wi)                         override;
            float pdf       (const shading_data& sd, const vec3& wi) override;

            [[nodiscard]]
            bool  is_area               () const override;
            float total_integrated_power() override;

            vec3 normal(const shading_data& sd, const vec3& wi) override;

            vec3 sample_position_from_light (sampler* s, float& pdf) override;
            vec3 sample_direction_from_light(sampler* s, float& pdf) override;
        };

        class environment_map : public base
        {
        public:
            texture* env{nullptr};

            explicit environment_map(texture* _env) noexcept;

            vec3 sample     (const shading_data& sd, sampler* s,
                             color& emitted_colour, float& pdf)      override;
            color evaluate  (const vec3& wi)                         override;
            float pdf       (const shading_data& sd, const vec3& wi) override;

            [[nodiscard]]
            bool  is_area               () const override;
            float total_integrated_power() override;

            vec3 normal(const shading_data& sd, const vec3& wi) override;

            vec3 sample_position_from_light (sampler* s, float& pdf) override;
            vec3 sample_direction_from_light(sampler* s, float& pdf) override;
        };
    }

} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_LIGHTS_H
