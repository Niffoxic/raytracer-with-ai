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
#ifndef RAYTRACER_WITH_AI_RENDERER_H
#define RAYTRACER_WITH_AI_RENDERER_H

#include "framework/core.h"
#include "denoiser.h"
#include "framework/geometry.h"
#include "framework/materials.h"
#include "sampler/sampling.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace GamesEngineeringBase { class Window; }

namespace fox_tracer
{
    namespace scene
    {
        class container;
    }

    struct rt_config;

    class film;
    class adaptive_sampler;
}

namespace fox_tracer::render
{
    class light_tracer;
    class photon_map;
    class photon_shooter;
    class photon_gatherer;
    class instant_radiosity;

    namespace renderer_config
    {
        inline constexpr int merge_strips = 64;
    }

    class ray_tracer
    {
    public:
        enum class pool_phase : int
        {
            tile_render    = 0,
            light_paths    = 1,
            tile_tonemap   = 2,
            vpl_shoot      = 3,
            tile_vpl       = 4,
            photon_shoot_g = 5,
            photon_shoot_c = 6,
            tile_pm        = 7
        };

        scene::container*             target_scene{nullptr};
        GamesEngineeringBase::Window* canvas{nullptr};
        std::unique_ptr<film>         render_film;
        std::unique_ptr<light_tracer> lt;

        std::unique_ptr<photon_map>      pm_global;
        std::unique_ptr<photon_map>      pm_caustic;
        std::unique_ptr<photon_shooter>  photon_shoot;
        std::unique_ptr<photon_gatherer> photon_gather;

        std::unique_ptr<instant_radiosity> ir;
        std::unique_ptr<adaptive_sampler> adaptive;

        std::vector<std::unique_ptr<sampler>> samplers;
        int                                   num_procs{1};

         ray_tracer() noexcept;
        ~ray_tracer();

        ray_tracer(const ray_tracer&)            = delete;
        ray_tracer& operator=(const ray_tracer&) = delete;
        ray_tracer(ray_tracer&&)                 = delete;
        ray_tracer& operator=(ray_tracer&&)      = delete;

        void init(scene::container* _scene, GamesEngineeringBase::Window* _canvas);

        void rebind_scene(scene::container* _scene);

        void clear();
        void reset();

        [[nodiscard]] bool reached_target() const;
        [[nodiscard]] int  get_spp       () const;
        [[nodiscard]] int  get_tile_count() const;

        void start();
        void stop ();

        void save_hdr(const std::string& filename) const;
        void save_png(const std::string& filename) const;

        enum class denoise_status : int
        {
            idle    = 0,
            running = 1,
            ok      = 2,
            failed  = 3
        };

        bool denoise_async();
        void poll_denoise ();

        [[nodiscard]] denoise_status current_denoise_status() const noexcept
        {
            return denoise_status_.load(std::memory_order_acquire);
        }
        [[nodiscard]] float last_denoise_ms       () const  noexcept { return denoise_last_ms_; }
        [[nodiscard]] bool  display_denoised      () const  noexcept { return display_denoised_; }
                      void  clear_denoised_display()        noexcept { display_denoised_ = false; }

        [[nodiscard]] std::string   last_denoise_message();
        [[nodiscard]] bool          denoiser_available  ();

        void set_canvas_locked(bool v) noexcept
        {
            canvas_locked_.store(v, std::memory_order_release);
        }

        [[nodiscard]] bool canvas_locked() const noexcept
        {
            return canvas_locked_.load(std::memory_order_acquire);
        }

        denoiser& get_denoiser() noexcept { return denoiser_; }

        color compute_direct(const shading_data& sd, sampler* s, bool use_mis = false);

        color path_trace    (geometry::ray r, sampler* s);
        color direct_only   (geometry::ray& r, sampler* s);
        color albedo        (geometry::ray& r);
        color view_normals  (geometry::ray& r);

        color evaluate_background(const geometry::ray& r) const;

    private:
        void build_tile_order           ();
        void refresh_filter_if_needed   ();
        void rebuild_samplers_if_needed ();
        void refresh_adaptive_if_needed () const;
        void refresh_adaptive_budget    ();

        [[nodiscard]] int adaptive_samples_for(int x, int y,
                                               bool adaptive_active,
                                               int  max_per_pixel) const noexcept;

        static std::uint32_t seed_for(std::uint32_t ep, int spp, int thread_id) noexcept;

        void dispatcher_loop();
        void run_one_pass   ();

        color sample_for_mode(geometry::ray& r, sampler* s, int mode);

        void shade_tile(unsigned int x0, unsigned int y0,
                        unsigned int x1, unsigned int y1,
                        sampler* s, std::uint32_t ep);

        void shade_tile_light(unsigned int x0, unsigned int y0,
                              unsigned int x1, unsigned int y1,
                              sampler* s, std::uint32_t ep);

        void shoot_photons_chunk(unsigned int n_paths,
                                 sampler* s, bool for_caustic,
                                 std::uint32_t ep,
                                 int worker_id);

        void shade_tile_pm(unsigned int x0, unsigned int y0,
                           unsigned int x1, unsigned int y1,
                           sampler* s, std::uint32_t ep);

        void shoot_vpls_chunk(unsigned int n_paths,
                              sampler* s, std::uint32_t ep,
                              int worker_id);

        void shade_tile_vpl(unsigned int x0, unsigned int y0,
                            unsigned int x1, unsigned int y1,
                            sampler* s, std::uint32_t ep);

        void tonemap_tile(unsigned int x0, unsigned int y0,
                          unsigned int x1, unsigned int y1);

        void merge_scratch(color* scratch,
                           int buf_w, int buf_h,
                           int buf_x0, int buf_y0);

        void shutdown_pool   ();
        void run_pass_on_pool();
        void do_denoise_work ();

    private:
        std::atomic<bool>          stop_flag{false};
        std::atomic<bool>          started{false};
        std::atomic<bool>          pause_requested{false};
        std::atomic<std::uint32_t> epoch{0};
        std::thread                dispatcher_thread;
        std::mutex                 dispatch_mtx;
        std::condition_variable    dispatch_cv;
        bool                       dispatcher_paused{false};

        std::array<std::mutex, renderer_config::merge_strips> merge_stripes;

        std::vector<int> tile_order;
        int              tile_size  {64};
        int              tiles_x    {0};
        int              tiles_y    {0};

        std::uint32_t    cached_filter_gen  {0};
        std::uint32_t    cached_sampler_gen {0};

        std::vector<std::thread> pool_threads;
        std::mutex               pool_mtx;
        std::condition_variable  pool_cv_start;
        std::condition_variable  pool_cv_done;
        std::atomic<int>         pool_next_tile     {0};
        std::atomic<int>         pool_done_counter  {0};
        std::atomic<std::uint32_t> pool_work_gen    {0};
        std::uint32_t            pool_last_seen_gen {0};
        std::atomic<bool>        pool_shutdown      {false};
        std::uint32_t            pool_pass_epoch    {0};
        std::atomic<int>         pool_phase_mode{
            static_cast<int>(pool_phase::tile_render)
        };

        denoiser                     denoiser_;
        bool                         display_denoised_{false};
        std::atomic<bool>            canvas_locked_{false};

        std::atomic<denoise_status>  denoise_status_{denoise_status::idle};
        std::thread                  denoise_worker_;
        std::mutex                   denoise_msg_mtx_;
        std::string                  denoise_message_;
        float                        denoise_last_ms_{0.0f};
    };

} // fox_tracer::render

#endif //RAYTRACER_WITH_AI_RENDERER_H
