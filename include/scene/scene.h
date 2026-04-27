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
#ifndef RAYTRACER_WITH_AI_SCENE_H
#define RAYTRACER_WITH_AI_SCENE_H

#include <map>

#include "framework/core.h"
#include "framework/geometry.h"
#include "framework/materials.h"
#include "framework/lights.h"

#include <memory>
#include <shared_mutex>
#include <vector>

namespace fox_tracer
{
    class sampler;
    class shading_data;
    class texture_cache;

    class camera
    {
    public:
        matrix   projection_matrix;
        matrix   inverse_projection_matrix;
        matrix   camera_to_world;
        matrix   camera_to_view;
        matrix   world_to_clip;

        vec3 origin;
        vec3 view_direction;
        vec3 forward_ws;
        vec3 right_ws;
        vec3 up_ws;

        float    a_film         { 0.0f };
        float    width          { 0.0f };
        float    height         { 0.0f };
        float    lens_radius    { 0.0f };
        float    focal_distance { 1.0f };
        bool     flip_x         { false };
        float    fov_deg        { 45.0f };

        void init       (const matrix& projection, int screen_width, int screen_height);
        void update_view(const matrix& V);

        void set_thin_lens(float r_lens, float f_dist) noexcept;
        void apply_intrinsics(float new_fov_deg) noexcept;

        [[nodiscard]] geometry::ray generate_ray          (float x, float y) const;
        [[nodiscard]]geometry:: ray generate_ray_thin_lens(float x, float y,
                                                            sampler* s)      const;

        bool project_onto_camera(const vec3& p, float& x, float& y) const;

    private:
        void update_basis_cache() noexcept;
    };

    class texture_cache
    {
    public:
        texture_cache() = default;
        ~texture_cache();

        texture_cache(const texture_cache&)            = delete;
        texture_cache& operator=(const texture_cache&) = delete;

        // Returns a borrowed pointer; the cache retains ownership.
        texture* get_or_load(const std::string& filename);

    private:
        std::map<std::string, texture*> textures_;
        std::shared_mutex               mtx_;
    };

    namespace scene
    {
        class container
        {
        public:
            std::vector<geometry::triangle>  triangles;
            std::vector<bsdf::base*>         materials;
            std::vector<lights::base*>       lights;
            lights::base*                    background{ nullptr };
            accelerated_structure::bvh_node* bvh       { nullptr };
            camera                           cam;
            geometry::aabb                   bounds;
            std::vector<float>               light_power_cdf;
            std::vector<int>                 triangle_to_light;
            int                              background_light_idx{ -1 };

            std::unique_ptr<texture_cache> textures;

             container() noexcept = default;
            ~container();

            container(const container&)            = delete;
            container& operator=(const container&) = delete;
            container(container&&)                 = delete;
            container& operator=(container&&)      = delete;

            void init(const std::vector<geometry::triangle>& mesh_triangles,
                      const std::vector<bsdf::base*>&        mesh_materials,
                      lights::base*                          _background);

            void build();

            [[nodiscard]] accelerated_structure::intersection_data traverse(
                const geometry::ray& r) const;

            [[nodiscard]] lights::base* sample_light(sampler* s, float& pmf) const;

            [[nodiscard]] bool visible(const vec3& p1,
                                       const vec3& p2) const;

            [[nodiscard]] bool visible(const vec3& p1, const vec3& g_n1,
                                       const vec3& p2, const vec3& g_n2) const;

            [[nodiscard]] color emit(const geometry::triangle* light_tri,
                                     const shading_data& sd,
                                     const vec3& wi)     const;
            [[nodiscard]] float light_pmf_by_index(int light_idx) const noexcept;

            [[nodiscard]] shading_data calculate_shading_data(
                const accelerated_structure::intersection_data& intersection,
                const geometry::ray& r) const;
        };
    }

} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_SCENE_H
