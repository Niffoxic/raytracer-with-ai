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
#ifndef RAYTRACER_WITH_AI_GEOMETRY_H
#define RAYTRACER_WITH_AI_GEOMETRY_H

#include "core.h"
#include <vector>

namespace fox_tracer
{
    class sampler;
    namespace geometry
    {
        class ray
        {
        public:
            vec3 o;
            vec3 dir;
            vec3 inv_dir;

            ray() noexcept = default;
            ray(const vec3& _o, const vec3& _d) noexcept;

            void init(const vec3& _o, const vec3& _d) noexcept;

            [[nodiscard]] vec3 at(float t) const noexcept;
        };

        class plane
        {
        public:
            vec3 n;
            float    d{};

            void init(const vec3& _n, float _d) noexcept;
            bool ray_intersect(const ray& r, float& t) const noexcept;
        };

        class triangle
        {
        public:
            vertex   vertices[3];
            vec3     e1;               // v2 - v1
            vec3     e2;               // v0 - v2
            // Pre-computed edges used by ray_intersect (Möller-Trumbore).
            // Storing them avoids recomputing v1-v0 and v2-v0 on every ray hit.
            vec3     e0p;              // v1 - v0
            vec3     e1p;              // v2 - v0
            vec3     n;
            float    area{};
            float    d{};
            unsigned int material_index{};

            void init(const vertex& v0, const vertex& v1, const vertex& v2,
                      unsigned int _material_index) noexcept;

            [[nodiscard]] vec3 centre() const noexcept;

            bool ray_intersect(const ray& r, float& t, float& u, float& v) const noexcept;

            void interpolate_attributes(float alpha, float beta, float gamma,
                                        vec3& interpolated_normal,
                                        float& interpolated_u,
                                        float& interpolated_v) const noexcept;

            vec3 sample(sampler* s, float& pdf) const noexcept;
            [[nodiscard]] vec3 g_normal() const noexcept;
        };

        class aabb
        {
        public:
            vec3 max;
            vec3 min;

            aabb() noexcept;

            void reset() noexcept;
            void extend(const vec3& p) noexcept;

            bool ray_aabb(const ray& r, float& t) const noexcept;
            [[nodiscard]] bool ray_aabb(const ray& r) const noexcept;

            [[nodiscard]] float area() const noexcept;
        };

        class sphere
        {
        public:
            vec3 centre;
            float    radius{};

            void init(const vec3& _centre, float _radius) noexcept;

            bool ray_intersect(const ray& r, float& t) const noexcept;
        };
    } // namespace geometry
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_GEOMETRY_H
