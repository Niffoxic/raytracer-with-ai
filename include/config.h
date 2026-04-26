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
#ifndef RAYTRACER_WITH_AI_CONFIG_H
#define RAYTRACER_WITH_AI_CONFIG_H

#include <atomic>
#include <cstdint>

namespace fox_tracer
{
    enum class hemisphere_sampling
    {
        cosine_weighted = 0,
        uniform         = 1
    };

    enum class light_pick
    {
        uniform        = 0,
        power_weighted = 1
    };

    struct rt_defaults
    {
        static constexpr int  hemisphere_mode  = static_cast<int>(hemisphere_sampling::cosine_weighted);
        static constexpr int  light_pick_mode  = static_cast<int>(light_pick::uniform);
        static constexpr bool use_bvh = true;

        //~ Camera
        static constexpr float fov               = 45.0f;
        static constexpr float move_speed        = 1.0f;
        static constexpr float mouse_sensitivity = 0.0025f;
        static constexpr float lens_radius       = 0.0f;
        static constexpr float focal_distance    = 1.0f;

        //~ assets
        static constexpr bool normalize_obj      = true;
        static constexpr float normalize_obj_max = 1.0f;

        //~ UI Properties
        static constexpr bool pause_render = false;
    };

    struct rt_config
    {
        std::atomic<int>  hemisphere_mode   { rt_defaults::hemisphere_mode };
        std::atomic<bool> use_bvh           { rt_defaults::use_bvh };
        std::atomic<int>  light_pick_mode   { rt_defaults::light_pick_mode };

        //~ camera settings
        std::atomic<float> fov              {rt_defaults::fov};
        std::atomic<float> move_speed       {rt_defaults::move_speed};
        std::atomic<float> mouse_sensitivity{rt_defaults::mouse_sensitivity};
        std::atomic<float> lens_radius      {rt_defaults::lens_radius};
        std::atomic<float> focal_distance   {rt_defaults::focal_distance};

        //~ assets
        std::atomic<bool>  normalize_obj    {rt_defaults::normalize_obj};
        std::atomic<float> normalize_obj_max{rt_defaults::normalize_obj_max};

        //~ properties
        std::atomic<std::uint32_t>   reset_generation{0};
        std::atomic<bool>           pause_render{rt_defaults::pause_render};

        void request_reset() noexcept
        {
            reset_generation.fetch_add(1, std::memory_order_release);
        }
    };

    inline rt_config& config() noexcept
    {
        static rt_config c;
        return c;
    }
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_CONFIG_H
