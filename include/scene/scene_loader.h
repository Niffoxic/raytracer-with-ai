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
#ifndef RAYTRACER_WITH_AI_SCENE_LOADER_H
#define RAYTRACER_WITH_AI_SCENE_LOADER_H

#include "framework/core.h"
#include "utils/assets_loader.h"
#include "scene/scene.h"

#include <functional>
#include <string>
#include <unordered_map>

namespace fox_tracer
{
    class texture;
    namespace bsdf
    {
        class base;
    }
    namespace scene
    {
        class container;
    }
    class camera;

    class rt_camera
    {
    public:
        vec3 from;
        vec3 to;
        vec3 up     {0.0f, 1.0f, 0.0f};
        camera*  cam{ nullptr };

        float movespeed {1.0f};
        float rotspeed  {5.0f};

        float yaw  {0.0f};
        float pitch{0.0f};

        bool  initialized{false};

        void bind_to_scene(camera* c, const vec3& _from,
                           const vec3& _to, const vec3& _up);

        void forward        (float dt_scale = 1.0f);
        void back           (float dt_scale = 1.0f);
        void strafe_left    (float dt_scale = 1.0f);
        void strafe_right   (float dt_scale = 1.0f);
        void fly_up         (float dt_scale = 1.0f);
        void fly_down(      float dt_scale = 1.0f);

        void left       ();
        void right      ();
        void apply_view ();

        void look(int dx_px, int dy_px);
    };

    rt_camera& view_camera() noexcept;

    namespace bsdf
    {
        class factory
        {
        public:
            using creator_fn = std::function<base*(const loader::gem_material&,
                                                   const std::string& scene_dir,
                                                   texture_cache&     tex_cache)>;

            static factory& global() noexcept;

            void register_creator(const std::string& name, creator_fn fn);

            base* create(const std::string&          name,
                         const loader::gem_material& mat,
                         const std::string&          scene_dir,
                         texture_cache&              tex_cache) const;

            factory(const factory&)            = delete;
            factory& operator=(const factory&) = delete;
        private:
            std::unordered_map<std::string, creator_fn> creators_;

            factory() = default;
            static void install_defaults(factory& f);
        };
    } // namespace bsdf

    scene::container* build_scene_from_gem( loader::gem_scene& gemscene,
                                            const std::string& scene_name,
                                            int override_w = 0,
                                            int override_h = 0);

    scene::container* load_scene(const std::string& scene_name,
                      int override_w = 0, int override_h = 0);
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_SCENE_LOADER_H
