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
#ifndef RAYTRACER_WITH_AI_MATERIALS_H
#define RAYTRACER_WITH_AI_MATERIALS_H

#include "core.h"

#pragma warning( disable : 4244)
#pragma warning( disable : 4305)



namespace fox_tracer
{
    class texture;
    class sampler;

    namespace bsdf
    {
        class base;
    }

    class shading_data
    {
    public:
        vec3 x;
        vec3 wo;
        vec3 s_normal;
        vec3 g_normal;

        float tu{};
        float tv{};

        frame       shading_frame;
        bsdf::base* surface_bsdf{nullptr};
        float       t{};

        shading_data() noexcept = default;
        shading_data(const vec3& _x, const vec3& n) noexcept;
    };

    namespace bsdf
    {
        class base
        {
        public:
            color emission;

            virtual ~base() = default;

            virtual vec3 sample(const shading_data& sd, sampler* s,
                                color& reflected_colour, float& pdf) = 0;

            virtual color evaluate  (const shading_data& sd, const vec3& wi) = 0;
            virtual float pdf       (const shading_data& sd, const vec3& wi) = 0;
            virtual float mask      (const shading_data& sd) = 0;

            [[nodiscard]] virtual bool is_pure_specular () const = 0;
            [[nodiscard]] virtual bool is_two_sided     () const = 0;

            [[nodiscard]] bool  is_light() const noexcept;
            [[nodiscard]] color emit(const shading_data& sd,
                                     const vec3& wi) const noexcept;

            void add_light(const color& _emission) noexcept;
        };

        class diffuse final: public base
        {
        public:
            texture* albedo{nullptr};

            diffuse() = default;
            explicit diffuse(texture* _albedo) noexcept;

            vec3 sample(const shading_data& sd, sampler* s,
                            color& reflected_colour, float& pdf) override;

            color evaluate  (const shading_data& sd, const vec3& wi) override;
            float pdf       (const shading_data& sd, const vec3& wi) override;
            float mask      (const shading_data& sd)                 override;

            [[nodiscard]] bool is_pure_specular () const override;
            [[nodiscard]] bool is_two_sided     () const override;
        };

        class mirror final: public base
        {
        public:
            texture* albedo{nullptr};

            mirror() = default;
            explicit mirror(texture* _albedo) noexcept;

            vec3 sample(const shading_data& sd, sampler* s,
                            color& reflected_colour, float& pdf) override;

            color evaluate  (const shading_data& sd, const vec3& wi) override;
            float pdf       (const shading_data& sd, const vec3& wi) override;
            float mask      (const shading_data& sd)                 override;

            [[nodiscard]] bool is_pure_specular () const override;
            [[nodiscard]] bool is_two_sided     () const override;
        };

        class conductor final: public base
        {
        public:
            texture* albedo{nullptr};
            color    eta;
            color    k;
            float    alpha{};

            conductor() = default;
            conductor(texture* _albedo, const color& _eta, const color& _k,
                           float roughness) noexcept;

            vec3 sample(const shading_data& sd, sampler* s,
                            color& reflected_colour, float& pdf) override;

            color evaluate  (const shading_data& sd, const vec3& wi) override;
            float pdf       (const shading_data& sd, const vec3& wi) override;
            float mask      (const shading_data& sd)                 override;

            [[nodiscard]] bool is_pure_specular () const override;
            [[nodiscard]] bool is_two_sided     () const override;
        };

        class glass final: public base
        {
        public:
            texture* albedo{nullptr};
            float    int_ior{};
            float    ext_ior{};

            glass() = default;
            glass(texture* _albedo, float _int_ior, float _ext_ior) noexcept;

            vec3 sample(const shading_data& sd, sampler* s,
                            color& reflected_colour, float& pdf) override;

            color evaluate  (const shading_data& sd, const vec3& wi) override;
            float pdf       (const shading_data& sd, const vec3& wi) override;
            float mask      (const shading_data& sd)                 override;

            [[nodiscard]] bool is_pure_specular () const override;
            [[nodiscard]] bool is_two_sided     () const override;
        };

        class dielectric final: public base
        {
        public:
            texture* albedo{nullptr};
            float    int_ior{};
            float    ext_ior{};
            float    alpha  {};

            dielectric() = default;
            dielectric(texture* _albedo, float _int_ior, float _ext_ior,
                            float roughness) noexcept;

            vec3 sample(const shading_data& sd, sampler* s,
                            color& reflected_colour, float& pdf) override;

            color evaluate  (const shading_data& sd, const vec3& wi) override;
            float pdf       (const shading_data& sd, const vec3& wi) override;
            float mask      (const shading_data& sd)                 override;

            [[nodiscard]] bool is_pure_specular () const override;
            [[nodiscard]] bool is_two_sided     () const override;
        };

        class oren_nayar final: public base
        {
        public:
            texture* albedo{nullptr};
            float    sigma {};

            oren_nayar() = default;
            oren_nayar(texture* _albedo, float _sigma) noexcept;

            vec3 sample(const shading_data& sd, sampler* s,
                            color& reflected_colour, float& pdf) override;

            color evaluate  (const shading_data& sd, const vec3& wi) override;
            float pdf       (const shading_data& sd, const vec3& wi) override;
            float mask      (const shading_data& sd)                 override;

            [[nodiscard]] bool is_pure_specular () const override;
            [[nodiscard]] bool is_two_sided     () const override;
        };

        class plastic final: public base
        {
        public:
            texture* albedo{nullptr};
            float    int_ior{};
            float    ext_ior{};
            float    alpha  {};

            plastic() = default;
            plastic(texture* _albedo, float _int_ior, float _ext_ior,
                         float roughness) noexcept;

            [[nodiscard]] float alpha_to_phong_exponent() const noexcept;

            vec3 sample(const shading_data& sd, sampler* s,
                            color& reflected_colour, float& pdf) override;

            color evaluate  (const shading_data& sd, const vec3& wi) override;
            float pdf       (const shading_data& sd, const vec3& wi) override;
            float mask      (const shading_data& sd)                 override;

            [[nodiscard]] bool is_pure_specular () const override;
            [[nodiscard]] bool is_two_sided     () const override;
        };

        namespace fresnel
        {
            float dielectric(float cos_theta, float ior_int, float ior_ext) noexcept;
            color conductor (float cos_theta, const color& ior, const color& k) noexcept;
        }

        namespace ggx
        {
            float lambda(const vec3& wi, float alpha) noexcept;
            float g     (const vec3& wi, const vec3& wo, float alpha) noexcept;
            float d     (const vec3& h, float alpha) noexcept;
        }
    } // namespace bsdf

}

#endif //RAYTRACER_WITH_AI_MATERIALS_H
