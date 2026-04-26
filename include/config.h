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
    enum class render_mode
    {
        path_trace  = 0,
        direct      = 1,
        albedo      = 2,
        normals     = 3,
        vpl         = 4,
        photon_map  = 5
    };

    enum class pixel_filter_kind
    {
        mitchell      = 0,
        gaussian      = 1,
        box           = 2,
        triangle      = 3,
        lanczos_sinc  = 4
    };

    enum class sampling_technique
    {
        bsdf_only = 0,
        nee       = 1,
        mis       = 2
    };

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
        // Path tracer
        static constexpr int  max_depth        = 16;
        static constexpr int  rr_depth         = 3;
        static constexpr int  sampling_tech    =
            static_cast<int>(sampling_technique::mis);
        static constexpr int  hemisphere_mode  =
            static_cast<int>(hemisphere_sampling::cosine_weighted);
        static constexpr int  light_pick_mode  =
            static_cast<int>(light_pick::uniform);
        static constexpr bool use_rr           = true;
        static constexpr int  render_mode_v    = static_cast<int>(render_mode::path_trace);

        // Dispatcher
        static constexpr int samples_per_call = 16;
        static constexpr int target_spp       = 8192;
        static constexpr int tile_size        = 64;
        static constexpr int num_threads      = 0;

        // Camera
        static constexpr float fov               = 45.0f;
        static constexpr float move_speed        = 1.0f;
        static constexpr float mouse_sensitivity = 0.0025f;
        static constexpr float lens_radius       = 0.0f;
        static constexpr float focal_distance    = 1.0f;

        static constexpr float exposure   = 1.0f;
        static constexpr float gamma      = 2.2f;
        static constexpr float contrast   = 1.0f;
        static constexpr float saturation = 1.0f;

        static constexpr float mitchell_b = 1.0f / 3.0f;
        static constexpr float mitchell_c = 1.0f / 3.0f;
        static constexpr float gaussian_alpha  = 2.0f;

        static constexpr float box_radius_x      = 0.5f;
        static constexpr float box_radius_y      = 0.5f;
        static constexpr float mitchell_radius_x = 2.0f;
        static constexpr float mitchell_radius_y = 2.0f;
        static constexpr float gaussian_radius_x = 1.5f;
        static constexpr float gaussian_radius_y = 1.5f;
        static constexpr float triangle_radius_x = 2.0f;
        static constexpr float triangle_radius_y = 2.0f;
        static constexpr float lanczos_radius_x  = 2.0f;
        static constexpr float lanczos_radius_y  = 2.0f;
        static constexpr float lanczos_tau       = 3.0f;

        // Background override
        static constexpr bool  override_background = false;
        static constexpr float bg_r = 0.0f;
        static constexpr float bg_g = 0.0f;
        static constexpr float bg_b = 0.0f;

        // UI state
        static constexpr bool show_ui      = true;
        static constexpr bool pause_render = false;

        // Feature
        static constexpr bool use_bvh      = true;
        static constexpr int  pixel_filter = static_cast<int>(pixel_filter_kind::mitchell);
        static constexpr bool use_filter_importance_sampling = false;

        static constexpr bool  normalize_obj     = true;
        static constexpr float normalize_obj_max = 1.0f;

        // Photon Mapping
        static constexpr int   pm_p_global          = 200'000;
        static constexpr int   pm_p_caustic         =  50'000;
        static constexpr int   pm_k_global          = 100;
        static constexpr int   pm_k_caustic         = 60;
        static constexpr float pm_r_max_global      = 0.5f;
        static constexpr float pm_r_max_caustic     = 0.1f;
        static constexpr bool  pm_use_final_gather  = false;
        static constexpr int   pm_final_gather_rays = 16;

        // Instant Radiosity
        static constexpr int   ir_num_vpls      = 256;
        static constexpr float ir_min_dist_sq   = 1.0e-3f;
        static constexpr float ir_max_contrib   = 10.0f;

        static constexpr float pt_firefly_max_direct   = 5.0f;
        static constexpr float pt_firefly_max_indirect = 5.0f;

        static constexpr int  sampler_kind_v        = 0;   // mt_random
        static constexpr int  sampler_samples_axis  = 4;
        static constexpr bool sampler_scrambling    = true;
        static constexpr int  sampler_max_dims      = 256;

        static constexpr bool use_adaptive_sampling = true;
        static constexpr int  adaptive_block_size   = 16;
        static constexpr int  adaptive_warmup_spp   = 4;
        static constexpr int  adaptive_max_per_pixel = 2;
    };

    struct rt_config
    {
        std::atomic<int>  max_depth      {rt_defaults::max_depth};
        std::atomic<int>  rr_depth       {rt_defaults::rr_depth};
        std::atomic<int>  sampling_tech  {rt_defaults::sampling_tech};
        std::atomic<int>  hemisphere_mode{rt_defaults::hemisphere_mode};
        std::atomic<int>  light_pick_mode{rt_defaults::light_pick_mode};
        std::atomic<bool> use_rr         {rt_defaults::use_rr};
        std::atomic<int>  render_mode    {rt_defaults::render_mode_v};

        std::atomic<int> samples_per_call{rt_defaults::samples_per_call};
        std::atomic<int> target_spp      {rt_defaults::target_spp};
        std::atomic<int> tile_size       {rt_defaults::tile_size};
        std::atomic<int> num_threads     {rt_defaults::num_threads};

        std::atomic<float> fov              {rt_defaults::fov};
        std::atomic<float> move_speed       {rt_defaults::move_speed};
        std::atomic<float> mouse_sensitivity{rt_defaults::mouse_sensitivity};
        std::atomic<float> lens_radius      {rt_defaults::lens_radius};
        std::atomic<float> focal_distance   {rt_defaults::focal_distance};

        std::atomic<float> exposure   {rt_defaults::exposure};
        std::atomic<float> gamma      {rt_defaults::gamma};
        std::atomic<float> contrast   {rt_defaults::contrast};
        std::atomic<float> saturation {rt_defaults::saturation};

        std::atomic<float> mitchell_b{rt_defaults::mitchell_b};
        std::atomic<float> mitchell_c{rt_defaults::mitchell_c};

        std::atomic<float> gaussian_alpha {rt_defaults::gaussian_alpha};

        std::atomic<float> box_radius_x     {rt_defaults::box_radius_x};
        std::atomic<float> box_radius_y     {rt_defaults::box_radius_y};
        std::atomic<float> mitchell_radius_x{rt_defaults::mitchell_radius_x};
        std::atomic<float> mitchell_radius_y{rt_defaults::mitchell_radius_y};
        std::atomic<float> gaussian_radius_x{rt_defaults::gaussian_radius_x};
        std::atomic<float> gaussian_radius_y{rt_defaults::gaussian_radius_y};
        std::atomic<float> triangle_radius_x{rt_defaults::triangle_radius_x};
        std::atomic<float> triangle_radius_y{rt_defaults::triangle_radius_y};
        std::atomic<float> lanczos_radius_x {rt_defaults::lanczos_radius_x};
        std::atomic<float> lanczos_radius_y {rt_defaults::lanczos_radius_y};
        std::atomic<float> lanczos_tau      {rt_defaults::lanczos_tau};

        std::atomic<bool>  override_background{rt_defaults::override_background};
        std::atomic<float> bg_r{rt_defaults::bg_r};
        std::atomic<float> bg_g{rt_defaults::bg_g};
        std::atomic<float> bg_b{rt_defaults::bg_b};

        std::atomic<bool> show_ui     {rt_defaults::show_ui};
        std::atomic<bool> pause_render{rt_defaults::pause_render};

        std::atomic<bool> use_bvh     {rt_defaults::use_bvh};
        std::atomic<int>  pixel_filter{rt_defaults::pixel_filter};
        std::atomic<bool> use_filter_importance_sampling
            {rt_defaults::use_filter_importance_sampling};

        std::atomic<bool>  normalize_obj    {rt_defaults::normalize_obj};
        std::atomic<float> normalize_obj_max{rt_defaults::normalize_obj_max};

        std::atomic<int>   pm_p_global         {rt_defaults::pm_p_global};
        std::atomic<int>   pm_p_caustic        {rt_defaults::pm_p_caustic};
        std::atomic<int>   pm_k_global         {rt_defaults::pm_k_global};
        std::atomic<int>   pm_k_caustic        {rt_defaults::pm_k_caustic};
        std::atomic<float> pm_r_max_global     {rt_defaults::pm_r_max_global};
        std::atomic<float> pm_r_max_caustic    {rt_defaults::pm_r_max_caustic};
        std::atomic<bool>  pm_use_final_gather {rt_defaults::pm_use_final_gather};
        std::atomic<int>   pm_final_gather_rays{rt_defaults::pm_final_gather_rays};

        std::atomic<int>   ir_num_vpls   {rt_defaults::ir_num_vpls};
        std::atomic<float> ir_min_dist_sq{rt_defaults::ir_min_dist_sq};
        std::atomic<float> ir_max_contrib{rt_defaults::ir_max_contrib};

        std::atomic<float> pt_firefly_max_direct  {rt_defaults::pt_firefly_max_direct};
        std::atomic<float> pt_firefly_max_indirect{rt_defaults::pt_firefly_max_indirect};

        std::atomic<int>  sampler_kind        {rt_defaults::sampler_kind_v};
        std::atomic<int>  sampler_samples_axis{rt_defaults::sampler_samples_axis};
        std::atomic<bool> sampler_scrambling  {rt_defaults::sampler_scrambling};
        std::atomic<int>  sampler_max_dims    {rt_defaults::sampler_max_dims};

        std::atomic<bool> use_adaptive_sampling   {rt_defaults::use_adaptive_sampling};
        std::atomic<int>  adaptive_block_size     {rt_defaults::adaptive_block_size};
        std::atomic<int>  adaptive_warmup_spp     {rt_defaults::adaptive_warmup_spp};
        std::atomic<int>  adaptive_max_per_pixel  {rt_defaults::adaptive_max_per_pixel};

        std::atomic<std::uint32_t> filter_generation {0};
        std::atomic<std::uint32_t> sampler_generation{0};
        std::atomic<std::uint32_t> reset_generation  {0};

        void request_reset() noexcept
        {
            reset_generation.fetch_add(1, std::memory_order_release);
        }

        void set_max_depth(const int  v) noexcept
        {
            max_depth.store(v);
            request_reset();
        }

        void set_rr_depth(const int  v) noexcept
        {
            rr_depth.store(v);
            request_reset();
        }

        void set_sampling_tech  (const int  v) noexcept
        {
            sampling_tech.store(v);
            request_reset();
        }

        void set_hemisphere_mode(const int  v) noexcept
        {
            hemisphere_mode.store(v);
            request_reset();
        }

        void set_light_pick_mode(const int  v) noexcept
        {
            light_pick_mode.store(v);
            request_reset();
        }

        void set_use_rr(const bool v) noexcept
        {
            use_rr.store(v);
            request_reset();
        }

        void set_render_mode(const int  v) noexcept
        {
            render_mode.store(v);
            request_reset();
        }

        void set_fov(const float v) noexcept
        {
            fov.store(v);
            request_reset();
        }

        void set_lens_radius(const float v) noexcept
        {
            lens_radius.store(v);
            request_reset();
        }

        void set_focal_distance(const float v) noexcept
        {
            focal_distance.store(v);
            request_reset();
        }

        void set_override_background(const bool v) noexcept
        {
            override_background.store(v);
            request_reset();
        }

        void set_bg_r(const float v) noexcept
        {
            bg_r.store(v);
            request_reset();
        }

        void set_bg_g(const float v) noexcept
        {
            bg_g.store(v);
            request_reset();
        }

        void set_bg_b(const float v) noexcept
        {
            bg_b.store(v);
            request_reset();
        }

        void set_use_bvh(const bool v) noexcept
        {
            use_bvh.store(v);
            request_reset();
        }

        void set_tile_size(const int v) noexcept
        {
            tile_size.store(v);
            request_reset();
        }

        void set_pixel_filter(const int v) noexcept
        {
            pixel_filter.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_use_filter_importance_sampling(const bool v) noexcept
        {
            use_filter_importance_sampling.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_mitchell_b(const float v) noexcept
        {
            mitchell_b.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }
        void set_mitchell_c(const float v) noexcept
        {
            mitchell_c.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }
        void set_gaussian_alpha(const float v) noexcept
        {
            gaussian_alpha.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_box_radius_x(const float v) noexcept
        {
            box_radius_x.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_box_radius_y(const float v) noexcept
        {
            box_radius_y.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_mitchell_radius_x(const float v) noexcept
        {
            mitchell_radius_x.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_mitchell_radius_y(const float v) noexcept
        {
            mitchell_radius_y.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_gaussian_radius_x(const float v) noexcept
        {
            gaussian_radius_x.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_gaussian_radius_y(const float v) noexcept
        {
            gaussian_radius_y.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_triangle_radius_x(const float v) noexcept
        {
            triangle_radius_x.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_triangle_radius_y(const float v) noexcept
        {
            triangle_radius_y.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_lanczos_radius_x(const float v) noexcept
        {
            lanczos_radius_x.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_lanczos_radius_y(const float v) noexcept
        {
            lanczos_radius_y.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_lanczos_tau(const float v) noexcept
        {
            lanczos_tau.store(v);
            filter_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_pm_p_global(const int v) noexcept
        {
            pm_p_global.store(v);
            request_reset();
        }

        void set_pm_p_caustic(const int v) noexcept
        {
            pm_p_caustic.store(v);
            request_reset();
        }

        void set_pm_k_global(const int v) noexcept
        {
            pm_k_global.store(v);
            request_reset();
        }

        void set_pm_k_caustic(const int v) noexcept
        {
            pm_k_caustic.store(v);
            request_reset();
        }

        void set_pm_r_max_global(const float v) noexcept
        {
            pm_r_max_global.store(v);
            request_reset();
        }

        void set_pm_r_max_caustic(const float v) noexcept
        {
            pm_r_max_caustic.store(v);
            request_reset();
        }

        void set_pm_use_final_gather(const bool v) noexcept
        {
            pm_use_final_gather .store(v);
            request_reset();
        }

        void set_pm_final_gather_rays(const int v) noexcept
        {
            pm_final_gather_rays.store(v);
            request_reset();
        }

        void set_ir_num_vpls(const int v) noexcept
        {
            ir_num_vpls.store(v);
            request_reset();
        }

        void set_ir_min_dist_sq(const float v) noexcept
        {
            ir_min_dist_sq.store(v);
            request_reset();
        }

        void set_ir_max_contrib(const float v) noexcept
        {
            ir_max_contrib.store(v);
            request_reset();
        }

        void set_pt_firefly_max_direct(const float v) noexcept
        {
            pt_firefly_max_direct.store(v);
        }

        void set_pt_firefly_max_indirect(const float v) noexcept
        {
            pt_firefly_max_indirect.store(v);
        }

        void set_sampler_kind(const int v) noexcept
        {
            sampler_kind.store(v);
            sampler_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_sampler_samples_axis(const int v) noexcept
        {
            sampler_samples_axis.store(v);
            sampler_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_sampler_scrambling(const bool v) noexcept
        {
            sampler_scrambling.store(v);
            sampler_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_sampler_max_dims(const int v) noexcept
        {
            sampler_max_dims.store(v);
            sampler_generation.fetch_add(1, std::memory_order_release);
            request_reset();
        }

        void set_use_adaptive_sampling(const bool v) noexcept
        {
            use_adaptive_sampling.store(v);
        }

        void set_adaptive_block_size(const int v) noexcept
        {
            adaptive_block_size.store(v);
            request_reset();
        }

        void set_adaptive_warmup_spp(const int v) noexcept
        {
            adaptive_warmup_spp.store(v);
        }

        void set_adaptive_max_per_pixel(const int v) noexcept
        {
            adaptive_max_per_pixel.store(v);
        }
    };

    inline rt_config& config() noexcept
    {
        static rt_config c;
        return c;
    }
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_CONFIG_H
