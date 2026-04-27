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
#include "render/renderer.h"
#include "config.h"
#include "scene/scene.h"

#include "framework/imaging.h"
#include "framework/base.h"
#include "framework/lights.h"

#include "render/light_tracer.h"
#include "render/photon_map.h"
#include "render/photon_shooter.h"
#include "render/photon_gatherer.h"
#include "render/instant_radiosity.h"

#include "utils/helper.h"
#include "utils/logger.h"

#include "sampler/pixel_sampler.h"
#include "stb_image_write.h"

#include <windows.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <utility>

namespace fox_tracer::render
{
    ray_tracer::ray_tracer() noexcept = default;

    ray_tracer::~ray_tracer()
    {
        stop         ();
        shutdown_pool();

        {
            std::lock_guard<std::mutex> lk(dispatch_mtx);
            dispatch_cv.notify_all();
        }
        if (denoise_worker_.joinable())
        {
            denoise_worker_.join();
        }
    }

    void ray_tracer::init(
        scene::container* _scene,
        GamesEngineeringBase::Window* _canvas)
    {
        target_scene = _scene;
        canvas       = _canvas;

        if (target_scene != nullptr)
        {
            config().fov.store(target_scene->cam.fov_deg,
                std::memory_order_relaxed);
            target_scene->cam.set_thin_lens(
                std::max(0.0f,
                    config().lens_radius.load(std::memory_order_relaxed)
                    ),
                std::max(1.0e-4f,
                    config().focal_distance.load(std::memory_order_relaxed)
                    )
            );
        }

        render_film = std::make_unique<film>();
        render_film->init(
            static_cast<int>(_scene->cam.width),
            static_cast<int>(_scene->cam.height),
            filter::filter_factory(config().pixel_filter.load(std::memory_order_relaxed)));

        cached_filter_gen = config().filter_generation.load(std::memory_order_acquire);

        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        num_procs = static_cast<int>(sys_info.dwNumberOfProcessors);

        if (num_procs < 1) num_procs = 1;

        cached_sampler_gen = config().sampler_generation.load(std::memory_order_acquire);
        samplers.clear();
        rebuild_samplers_if_needed();

        tile_size = helper::clamp_tile_size(config().tile_size.load(std::memory_order_relaxed));
        tiles_x = (static_cast<int>(render_film->width)  + tile_size - 1) / tile_size;
        tiles_y = (static_cast<int>(render_film->height) + tile_size - 1) / tile_size;
        build_tile_order();

        adaptive = std::make_unique<adaptive_sampler>();
        adaptive->init(static_cast<int>(render_film->width),
                       static_cast<int>(render_film->height),
                       std::max(1, config().adaptive_block_size
                                       .load(std::memory_order_relaxed)));

        lt = std::make_unique<light_tracer>();
        lt->init(target_scene,
                 render_film.get(),
                 merge_stripes.data(),
                 merge_stripes.size(),
                 static_cast<std::size_t>(render_film->width) * render_film->height);

        pm_global  = std::make_unique<photon_map>();
        pm_caustic = std::make_unique<photon_map>();

        const float diag = (target_scene->bounds.max
                          - target_scene->bounds.min).length();

        const float safe_diag = (diag > 0.0f) ? diag : 1.0f;
        pm_global ->init(0.02f * safe_diag);
        pm_caustic->init(0.01f * safe_diag);

        pm_global ->resize_workers(num_procs);
        pm_caustic->resize_workers(num_procs);

        ir = std::make_unique<instant_radiosity>();
        ir->init(target_scene,
                 render_film.get(),
                 merge_stripes.data(),
                 merge_stripes.size(),
                 static_cast<std::size_t>(
                     config().ir_num_vpls.load(std::memory_order_relaxed)));
        ir->owner = this;
        ir->resize_workers(num_procs);

        photon_shoot = std::make_unique<photon_shooter>();
        photon_shoot->init(target_scene, pm_global.get(), pm_caustic.get());

        photon_gather = std::make_unique<photon_gatherer>();
        photon_gather->init(target_scene, pm_global.get(), pm_caustic.get());
        photon_gather->owner = this;

        pool_shutdown.store(false, std::memory_order_release);
        pool_work_gen.store(0, std::memory_order_release);
        pool_last_seen_gen = 0;
        pool_threads.clear();
        pool_threads.reserve(num_procs);
        for (int i = 0; i < num_procs; ++i)
        {
            pool_threads.emplace_back([this, i]()
            {
                std::uint32_t seen = 0;
                for (;;)
                {
                    {
                        std::unique_lock<std::mutex> lk(pool_mtx);
                        pool_cv_start.wait(lk, [&] {
                            return pool_shutdown.load(std::memory_order_acquire)
                                || pool_work_gen.load(std::memory_order_acquire) != seen;
                        });
                        if (pool_shutdown.load(std::memory_order_acquire)) return;
                        seen = pool_work_gen.load(std::memory_order_acquire);
                    }

                    const std::uint32_t ep    = pool_pass_epoch;
                    const int total_tiles    = tiles_x * tiles_y;
                    sampler* s = samplers[i].get();

                    const int hw_max = std::max(1, num_procs);
                    int active_now =
                        config().num_threads.load(std::memory_order_relaxed);
                    if (active_now <= 0)        active_now = hw_max;
                    if (active_now > hw_max)    active_now = hw_max;

                    const int phase = pool_phase_mode.load(std::memory_order_acquire);
                    while (i < active_now)
                    {
                        const int idx = pool_next_tile.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= total_tiles) break;
                        if (epoch.load(std::memory_order_acquire) != ep) break;
                        if (stop_flag.load(std::memory_order_acquire)) break;
                        if (pause_requested.load(std::memory_order_acquire)) break;

                        const int tile_id = tile_order[idx];
                        const int tx = tile_id % tiles_x;
                        const int ty = tile_id / tiles_x;
                        const auto x_start = static_cast<unsigned int>(tx * tile_size);
                        const auto y_start = static_cast<unsigned int>(ty * tile_size);
                        const auto x_end   = std::min<unsigned int>(x_start + tile_size,
                                                                            render_film->width);
                        const auto y_end   = std::min<unsigned int>(y_start + tile_size,
                                                                            render_film->height);

                        switch (static_cast<pool_phase>(phase))
                        {
                        case pool_phase::light_paths:
                            shade_tile_light(x_start, y_start, x_end, y_end, s, ep);
                            break;
                        case pool_phase::tile_tonemap:
                            tonemap_tile(x_start, y_start, x_end, y_end);
                            break;
                        case pool_phase::photon_shoot_g:
                        case pool_phase::photon_shoot_c:
                        {
                            const bool for_caustic =
                                (static_cast<pool_phase>(phase) == pool_phase::photon_shoot_c);
                            const std::size_t P = (photon_shoot == nullptr)
                                ? 0
                                : (for_caustic ? photon_shoot->p_caustic
                                               : photon_shoot->p_global);
                            const int total = std::max(1, tiles_x * tiles_y);
                            const std::size_t share = P / static_cast<std::size_t>(total);
                            const std::size_t rem   = P - share * static_cast<std::size_t>(total);
                            const std::size_t mine  = share
                                + ((static_cast<std::size_t>(idx) < rem) ? 1u : 0u);
                            shoot_photons_chunk(static_cast<unsigned int>(mine),
                                                s, for_caustic, ep, i);
                            break;
                        }
                        case pool_phase::tile_pm:
                            shade_tile_pm(x_start, y_start, x_end, y_end, s, ep);
                            break;
                        case pool_phase::vpl_shoot:
                        {
                            const int total  = tiles_x * tiles_y;
                            const int n_vpls = (ir != nullptr)
                                ? static_cast<int>(ir->num_vpls)
                                : 0;
                            const int share  = (total > 0) ? (n_vpls / total) : 0;
                            const int rem    = (total > 0) ? (n_vpls - share * total) : 0;
                            const int mine   = share + ((idx < rem) ? 1 : 0);
                            shoot_vpls_chunk(static_cast<unsigned int>(std::max(0, mine)),
                                             s, ep, i);
                            break;
                        }
                        case pool_phase::tile_vpl:
                            shade_tile_vpl(x_start, y_start, x_end, y_end, s, ep);
                            break;
                        case pool_phase::tile_render:
                        default:
                            shade_tile(x_start, y_start, x_end, y_end, s, ep);
                            break;
                        }
                    }

                    if (pool_done_counter.fetch_add(1, std::memory_order_acq_rel) + 1 == num_procs)
                    {
                        std::lock_guard<std::mutex> lk(pool_mtx);
                        pool_cv_done.notify_one();
                    }
                }
            });
        }

        clear();
    }

    void ray_tracer::rebind_scene(scene::container* _scene)
    {
        target_scene = _scene;
        if (lt)            lt->target_scene            = _scene;
        if (ir)            ir->target_scene            = _scene;
        if (photon_shoot)  photon_shoot->target_scene  = _scene;
        if (photon_gather) photon_gather->target_scene = _scene;
        if (render_film)   render_film->clear();
        if (canvas)        canvas->clear();
    }

    void ray_tracer::clear()
    {
        render_film->clear();
    }

    void ray_tracer::reset()
    {
        pause_requested.store(true, std::memory_order_release);
        epoch.fetch_add(1, std::memory_order_release);

        {
            std::unique_lock<std::mutex> lk(dispatch_mtx);
            dispatch_cv.notify_all();
            dispatch_cv.wait(lk, [&] {
                return !started.load() || dispatcher_paused;
            });
        }

        if (target_scene != nullptr)
        {
            target_scene->cam.apply_intrinsics(
                config().fov.load(std::memory_order_relaxed));
            target_scene->cam.set_thin_lens(
                std::max(0.0f,
                    config().lens_radius.load(std::memory_order_relaxed)),
                std::max(1.0e-4f,
                    config().focal_distance.load(std::memory_order_relaxed)));
        }

        refresh_filter_if_needed();
        rebuild_samplers_if_needed();
        refresh_adaptive_if_needed();

        const int desired_tile = helper::clamp_tile_size(
            config().tile_size.load(std::memory_order_relaxed));
        if (desired_tile != tile_size)
        {
            tile_size = desired_tile;
            tiles_x = (static_cast<int>(render_film->width)  + tile_size - 1) / tile_size;
            tiles_y = (static_cast<int>(render_film->height) + tile_size - 1) / tile_size;
            build_tile_order();
        }

        render_film->clear();
        if (canvas) canvas->clear();

        if (pm_global)  pm_global ->clear();
        if (pm_caustic) pm_caustic->clear();

        pause_requested.store(false, std::memory_order_release);
        {
            std::unique_lock<std::mutex> lk(dispatch_mtx);
            dispatch_cv.notify_all();
        }
    }

    void ray_tracer::refresh_filter_if_needed()
    {
        const std::uint32_t gen = config().filter_generation.load(std::memory_order_acquire);
        if (gen == cached_filter_gen) return;
        cached_filter_gen = gen;
        render_film->set_filter(filter::filter_factory(
            config().pixel_filter.load(std::memory_order_relaxed)));
    }

    void ray_tracer::rebuild_samplers_if_needed()
    {
        const std::uint32_t gen =
            config().sampler_generation.load(std::memory_order_acquire);
        if (gen == cached_sampler_gen && !samplers.empty()) return;
        cached_sampler_gen = gen;

        const auto kind = static_cast<sampling::sampler_kind>(
            config().sampler_kind.load(std::memory_order_relaxed));
        const int  spa  = std::max(1,
            config().sampler_samples_axis.load(std::memory_order_relaxed));
        const bool scr  = config().sampler_scrambling.load(std::memory_order_relaxed);
        const int  md   = std::max(1,
            config().sampler_max_dims.load(std::memory_order_relaxed));

        samplers.clear();
        samplers.reserve(num_procs);
        for (int i = 0; i < num_procs; ++i)
        {
            sampling::sampler_config cfg;
            cfg.kind             = kind;
            cfg.seed             = static_cast<unsigned int>(i + 1);
            cfg.samples_per_axis = spa;
            cfg.scrambling       = scr;
            cfg.max_dimensions   = md;
            samplers.push_back(make_sampler(cfg));
        }
    }

    void ray_tracer::refresh_adaptive_if_needed() const
    {
        if (adaptive == nullptr) return;
        const int desired_block = std::max(1,
            config().adaptive_block_size.load(std::memory_order_relaxed));
        const int film_w = static_cast<int>(render_film->width);
        const int film_h = static_cast<int>(render_film->height);
        if (adaptive->block_size == desired_block
            && adaptive->img_width  == film_w
            && adaptive->img_height == film_h)
        {
            return;
        }
        adaptive->init(film_w, film_h, desired_block);
    }

    void ray_tracer::refresh_adaptive_budget()
    {
        if (adaptive == nullptr) return;

        const int spp_now = render_film->SPP.load(std::memory_order_relaxed);
        const int warmup  = std::max(0,
            config().adaptive_warmup_spp.load(std::memory_order_relaxed));
        if (spp_now < warmup) return;

        const int max_per_pixel = std::max(1,
            config().adaptive_max_per_pixel.load(std::memory_order_relaxed));

        adaptive->compute_variance(*render_film);

        const int pixels_per_block = adaptive->block_size * adaptive->block_size;
        const int total_budget     = adaptive->img_width
                                   * adaptive->img_height
                                   * max_per_pixel;
        adaptive->allocate_samples(total_budget, pixels_per_block);
    }

    int ray_tracer::adaptive_samples_for(const int x, const int y,
                                         const bool adaptive_active,
                                         const int  max_per_pixel) const noexcept
    {
        if (!adaptive_active || adaptive == nullptr) return 1;
        const int b   = adaptive->block_index_for(x, y);
        const int per = adaptive->block_size * adaptive->block_size;
        const int n   = adaptive->samples_for_block(b) / std::max(1, per);
        return std::max(1, std::min(max_per_pixel, n));
    }

    bool ray_tracer::reached_target() const
    {
        const int effective = std::max(1, config().target_spp.load(std::memory_order_relaxed));
        return render_film->SPP.load(std::memory_order_relaxed) >= effective;
    }

    int ray_tracer::get_spp() const
    {
        return render_film->SPP.load(std::memory_order_relaxed);
    }

    int ray_tracer::get_tile_count() const
    {
        return tiles_x * tiles_y;
    }

    void ray_tracer::start()
    {
        bool expected = false;
        if (!started.compare_exchange_strong(expected, true)) return;
        stop_flag.store(false);
        dispatcher_thread = std::thread(&ray_tracer::dispatcher_loop, this);
    }

    void ray_tracer::stop()
    {
        if (!started.load()) return;
        stop_flag.store(true);
        {
            std::unique_lock<std::mutex> lk(dispatch_mtx);
            dispatch_cv.notify_all();
        }
        if (dispatcher_thread.joinable()) dispatcher_thread.join();
        started.store(false);
    }

    void ray_tracer::save_hdr(const std::string& filename) const
    {
        render_film->save(filename);
    }

    void ray_tracer::save_png(const std::string& filename) const
    {
        stbi_write_png(filename.c_str(),
                       canvas->getWidth(), canvas->getHeight(),
                       3, canvas->getBackBuffer(),
                       canvas->getWidth() * 3);
    }

    bool ray_tracer::denoiser_available()
    {
        return denoiser_.ensure_ready();
    }

    std::string ray_tracer::last_denoise_message()
    {
        std::lock_guard<std::mutex> lk(denoise_msg_mtx_);
        return denoise_message_;
    }

    bool ray_tracer::denoise_async()
    {
        if (denoise_status_.load(std::memory_order_acquire) == denoise_status::running)
            return false;
        if (target_scene == nullptr || render_film == nullptr || canvas == nullptr)
            return false;
        if (!denoiser_.ensure_ready()) return false;
        if (render_film->SPP.load(std::memory_order_relaxed) <= 0) return false;

        if (denoise_worker_.joinable())
        {
            denoise_worker_.join();
        }

        {
            std::lock_guard<std::mutex> lk(denoise_msg_mtx_);
            denoise_message_ = "Denoising...";
        }
        denoise_status_.store(denoise_status::running, std::memory_order_release);

        LOG_INFO("denoiser") << "denoise start: "
                             << render_film->width << "x" << render_film->height
                             << " spp=" << render_film->SPP.load(std::memory_order_relaxed);

        denoise_worker_ = std::thread([this]() { do_denoise_work(); });
        return true;
    }

    void ray_tracer::poll_denoise()
    {
        const auto st = denoise_status_.load(std::memory_order_acquire);
        if (st == denoise_status::running) return;
        if (denoise_worker_.joinable())
        {
            denoise_worker_.join();
        }
    }

    void ray_tracer::do_denoise_work()
    {
        const auto t0 = std::chrono::steady_clock::now();

        pause_requested.store(true, std::memory_order_release);
        epoch.fetch_add(1, std::memory_order_release);
        {
            std::unique_lock<std::mutex> lk(dispatch_mtx);
            dispatch_cv.notify_all();
            dispatch_cv.wait(lk, [&] {
                return !started.load() || dispatcher_paused;
            });
        }

        const int w = static_cast<int>(render_film->width);
        const int h = static_cast<int>(render_film->height);
        const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        const int spp_now = render_film->SPP.load(std::memory_order_relaxed);

        std::vector<color> colour_in (n);
        std::vector<color> albedo_in (n);
        std::vector<color> normal_in (n);
        std::vector<color> colour_out(n);

        const float inv_spp = 1.0f / static_cast<float>(std::max(1, spp_now));

        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const std::size_t idx = static_cast<std::size_t>(y) * w + x;
                colour_in[idx] = render_film->film_buffer[idx] * inv_spp;

                geometry::ray r = target_scene->cam.generate_ray(
                    static_cast<float>(x) + 0.5f,
                    static_cast<float>(y) + 0.5f);

                const accelerated_structure::intersection_data ix = target_scene->traverse(r);
                if (ix.t >= FLT_MAX)
                {
                    albedo_in[idx] = color(0.0f, 0.0f, 0.0f);
                    normal_in[idx] = color(0.0f, 0.0f, 0.0f);
                    continue;
                }

                const shading_data sd = target_scene->calculate_shading_data(ix, r);
                if (sd.surface_bsdf != nullptr && sd.surface_bsdf->is_light())
                {
                    albedo_in[idx] = color(1.0f, 1.0f, 1.0f);
                }
                else if (sd.surface_bsdf != nullptr)
                {
                    albedo_in[idx] = sd.surface_bsdf->albedo_color(sd);
                }
                else
                {
                    albedo_in[idx] = color(0.0f, 0.0f, 0.0f);
                }

                normal_in[idx] = color(sd.s_normal.x, sd.s_normal.y, sd.s_normal.z);
            }
        }

        const auto t_gbuf = std::chrono::steady_clock::now();
        LOG_INFO("denoiser") << "guides built ("
                             << std::chrono::duration_cast<std::chrono::milliseconds>(
                                    t_gbuf - t0).count()
                             << " ms), calling OIDN...";

        const bool ok = denoiser_.denoise(
            colour_in.data(),
            denoiser_.use_albedo ? albedo_in.data() : nullptr,
            denoiser_.use_normal ? normal_in.data() : nullptr,
            colour_out.data(),
            w, h,
            denoiser_.hdr_mode);

        if (ok)
        {
            tonemap_params tm;
            tm.exposure    = config().exposure  .load(std::memory_order_relaxed);
            tm.gamma       = std::max(0.01f, config().gamma.load(std::memory_order_relaxed));
            tm.contrast    = config().contrast  .load(std::memory_order_relaxed);
            tm.saturation  = config().saturation.load(std::memory_order_relaxed);

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    const color mapped = apply_tonemap(
                        colour_out[static_cast<std::size_t>(y) * w + x], tm);
                    const unsigned char r8 = static_cast<unsigned char>(
                        math::saturate(mapped.red) * 255.0f);
                    const unsigned char g8 = static_cast<unsigned char>(
                        math::saturate(mapped.green) * 255.0f);
                    const unsigned char b8 = static_cast<unsigned char>(
                        math::saturate(mapped.blue) * 255.0f);
                    canvas->draw(x, y, r8, g8, b8);
                }
            }

            display_denoised_ = true;
            config().pause_render.store(true, std::memory_order_relaxed);
        }

        pause_requested.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(dispatch_mtx);
            dispatch_cv.notify_all();
        }

        const auto t1 = std::chrono::steady_clock::now();
        const float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        denoise_last_ms_ = total_ms;

        char buf[160];
        if (ok)
        {
            std::snprintf(buf, sizeof(buf),
                          "Denoised %dx%d @ %d spp in %.0f ms (albedo=%d normal=%d hdr=%d)",
                          w, h, spp_now, total_ms,
                          denoiser_.use_albedo ? 1 : 0,
                          denoiser_.use_normal ? 1 : 0,
                          denoiser_.hdr_mode   ? 1 : 0);
            LOG_INFO("denoiser") << buf;
        }
        else
        {
            std::snprintf(buf, sizeof(buf),
                          "Denoise FAILED after %.0f ms (see log)", total_ms);
            LOG_ERROR("denoiser") << buf;
        }
        {
            std::lock_guard<std::mutex> lk(denoise_msg_mtx_);
            denoise_message_ = buf;
        }

        denoise_status_.store(ok ? denoise_status::ok : denoise_status::failed,
                              std::memory_order_release);
    }

    color ray_tracer::evaluate_background(const geometry::ray& r) const
    {
        if (config().override_background.load(std::memory_order_relaxed))
        {
            return {
                config().bg_r.load(std::memory_order_relaxed),
                config().bg_g.load(std::memory_order_relaxed),
                config().bg_b.load(std::memory_order_relaxed)};
        }
        return target_scene->background->evaluate(r.dir);
    }

    color ray_tracer::compute_direct(const shading_data& sd, sampler* s, bool use_mis)
    {
      if (sd.surface_bsdf->is_pure_specular())
        {
            return color(0.0f, 0.0f, 0.0f);
        }

        float light_pmf;
        lights::base* L_src = target_scene->sample_light(s, light_pmf);
        if (L_src == nullptr || light_pmf <= 0.0f) return {0.0f, 0.0f, 0.0f};

        color Le;
        float light_pdf;
        vec3 p = L_src->sample(sd, s, Le, light_pdf);

        if (light_pdf <= 0.0f || Le.luminance() <= 0.0f)
            return {0.0f, 0.0f, 0.0f};

        if (L_src->is_area())
        {
            vec3 wi = p - sd.x;
            const float dist2 = wi.length_squared();

            if (dist2 <= 0.0f)
                return {0.0f, 0.0f, 0.0f};

            const float inv_dist = 1.0f / std::sqrt(dist2);
            const float dist     = dist2 * inv_dist;
            wi = wi * inv_dist;

            const vec3 n_light = L_src->normal(sd, wi);
            const float cos_surface = math::dot(wi, sd.s_normal);
            const float cos_light   = -math::dot(wi, n_light);

            if (cos_surface <= 0.0f || cos_light <= 0.0f)
                return color(0.0f, 0.0f, 0.0f);

            if (!target_scene->visible(sd.x, sd.g_normal, p, n_light))
                return color(0.0f, 0.0f, 0.0f);

            const color f       = sd.surface_bsdf->evaluate(sd, wi);
            const float pdf_dir = light_pdf * light_pmf * (dist2 / cos_light);
            float w = 1.0f;
            if (use_mis)
            {
                const float bsdf_pdf = sd.surface_bsdf->pdf(sd, wi);
                w = helper::power_heuristic(pdf_dir, bsdf_pdf);
            }
            return f * Le * (cos_surface * w / pdf_dir);
        }

        const vec3 wi = p;
        const float cos_surface = math::dot(wi, sd.s_normal);
        if (cos_surface <= 0.0f) return color(0.0f, 0.0f, 0.0f);

        geometry::ray shadow;
        shadow.init(math::offset_ray_origin(sd.x, sd.g_normal, wi), wi);
        const accelerated_structure::intersection_data sh =
            target_scene->traverse(shadow);

        if (sh.t < FLT_MAX)
            return color(0.0f, 0.0f, 0.0f);

        const color f       = sd.surface_bsdf->evaluate(sd, wi);
        const float pdf_dir = light_pdf * light_pmf;
        float w = 1.0f;
        if (use_mis)
        {
            const float bsdf_pdf = sd.surface_bsdf->pdf(sd, wi);
            w = helper::power_heuristic(pdf_dir, bsdf_pdf);
        }
        return f * Le * (cos_surface * w / pdf_dir);

    }

    fox_tracer::color fox_tracer::render::ray_tracer::path_trace(
        geometry::ray r, sampler* s)
    {
        color L(0.0f, 0.0f, 0.0f);
        color path_throughput(1.0f, 1.0f, 1.0f);
        bool  count_emission = true;

        const int  max_depth = std::max(1, config().max_depth.load(std::memory_order_relaxed));
        const auto tech      = static_cast<sampling_technique>(
            config().sampling_tech.load(std::memory_order_relaxed));
        const bool use_nee   = tech != sampling_technique::bsdf_only;
        const bool use_mis   = tech == sampling_technique::mis;
        const bool use_rr    = config().use_rr.load(std::memory_order_relaxed);
        const int  rr_depth  = std::max(0, config().rr_depth.load(std::memory_order_relaxed));

        float prev_bsdf_pdf = 0.0f;
        bool  prev_specular = true;

        const int n_lights = static_cast<int>(target_scene->lights.size());

        const float firefly_max_direct_L   = std::max(0.0f,
            config().pt_firefly_max_direct  .load(std::memory_order_relaxed));
        const float firefly_max_indirect_L = std::max(0.0f,
            config().pt_firefly_max_indirect.load(std::memory_order_relaxed));
        auto firefly = [](color c, float max_L) -> color
        {
            if (!std::isfinite(c.red) ||
                !std::isfinite(c.green) ||
                !std::isfinite(c.blue))
            {
                return color(0.0f, 0.0f, 0.0f);
            }
            if (max_L <= 0.0f) return c;
            const float lum = c.luminance();
            if (lum > max_L) c = c * (max_L / lum);
            return c;
        };

        for (int depth = 0; depth < max_depth; ++depth)
        {
            const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
            const shading_data sd = target_scene->calculate_shading_data(intersection, r);

            if (sd.t >= FLT_MAX)
            {
                color bg = path_throughput * evaluate_background(r);
                const int bg_idx = target_scene->background_light_idx;
                if (use_mis && depth > 0 && !prev_specular && bg_idx >= 0)
                {
                    shading_data dummy;
                    const float light_pmf = target_scene->light_pmf_by_index(bg_idx);
                    const float bg_pdf    = target_scene->background->pdf(dummy, r.dir);
                    const float pdf_dir   = light_pmf * bg_pdf;
                    const float w         = helper::power_heuristic(prev_bsdf_pdf, pdf_dir);
                    bg = bg * w;
                }
                if (depth > 0) bg = firefly(bg, firefly_max_indirect_L);
                L = L + bg;
                return L;
            }

            if (sd.surface_bsdf->is_light())
            {
                if (count_emission)
                {
                    color em = path_throughput * sd.surface_bsdf->emit(sd, sd.wo);
                    if (use_mis && depth > 0 && !prev_specular && n_lights > 0)
                    {
                        const geometry::triangle& tri = target_scene->triangles[intersection.ID];
                        const float cos_lt  = -math::dot(r.dir, tri.g_normal());
                        const int   lidx    = (intersection.ID < target_scene->triangle_to_light.size())
                            ? target_scene->triangle_to_light[intersection.ID]
                            : -1;
                        if (cos_lt > 0.0f && tri.area > 0.0f && lidx >= 0)
                        {
                            const float dist      = intersection.t;
                            const float light_pmf = target_scene->light_pmf_by_index(lidx);
                            const float pdf_dir   = light_pmf
                                * (1.0f / tri.area)
                                * (dist * dist / cos_lt);
                            const float w = helper::power_heuristic(prev_bsdf_pdf, pdf_dir);
                            em = em * w;
                        }
                    }
                    if (depth > 0) em = firefly(em, firefly_max_indirect_L);
                    L = L + em;
                }
                return L;
            }

            if (use_nee)
            {
                color direct = path_throughput * compute_direct(sd, s, use_mis);
                direct = firefly(direct, firefly_max_direct_L);
                L = L + direct;
            }
            if (use_rr && depth >= rr_depth)
            {
                const float rr = std::min(0.95f,
                    std::max(path_throughput.red,
                    std::max(path_throughput.green, path_throughput.blue)));

                if (rr <= 0.0f) return L;
                if (s->next() > rr) return L;
                path_throughput = path_throughput * (1.0f / rr);
            }

            color bsdf_weight;
            float pdf;
            const vec3 wi = sd.surface_bsdf->sample(sd, s, bsdf_weight, pdf);
            if (pdf <= 0.0f || !std::isfinite(pdf)) return L;

            const float cos_theta = std::fabs(math::dot(wi, sd.s_normal));
            path_throughput = path_throughput * bsdf_weight * (cos_theta / pdf);

            if (!std::isfinite(path_throughput.red) ||
                !std::isfinite(path_throughput.green) ||
                !std::isfinite(path_throughput.blue)) return L;
            if (path_throughput.luminance() <= 0.0f) return L;

            if (use_mis)
            {
                count_emission = true;
            }
            else
            {
                count_emission = use_nee ? sd.surface_bsdf->is_pure_specular() : true;
            }
            prev_bsdf_pdf = pdf;
            prev_specular = sd.surface_bsdf->is_pure_specular();

            r.init(math::offset_ray_origin(sd.x, sd.g_normal, wi), wi);
        }
        return L;
    }

    color ray_tracer::direct_only(geometry::ray& r, sampler* s)
    {
        const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
        const shading_data sd = target_scene->calculate_shading_data(intersection, r);

        if (sd.t >= FLT_MAX) return evaluate_background(r);
        if (sd.surface_bsdf->is_light()) return sd.surface_bsdf->emit(sd, sd.wo);

        return compute_direct(sd, s);
    }

    color ray_tracer::albedo(geometry::ray& r)
    {
        const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
        const shading_data sd = target_scene->calculate_shading_data(intersection, r);

        if (sd.t < FLT_MAX)
        {
            if (sd.surface_bsdf->is_light())
                return sd.surface_bsdf->emit(sd, sd.wo);
            return sd.surface_bsdf->albedo_color(sd);
        }
        return evaluate_background(r);
    }

    color ray_tracer::view_normals(geometry::ray& r)
    {
        const accelerated_structure::intersection_data intersection = target_scene->traverse(r);
        if (intersection.t < FLT_MAX)
        {
            const shading_data sd = target_scene->calculate_shading_data(intersection, r);
            return color(std::fabs(sd.s_normal.x),
                         std::fabs(sd.s_normal.y),
                         std::fabs(sd.s_normal.z));
        }
        return color(0.0f, 0.0f, 0.0f);
    }

    void ray_tracer::build_tile_order()
    {
        auto morton_key = [](std::uint32_t x, std::uint32_t y) -> std::uint64_t
        {
            auto spread = [](std::uint32_t v) -> std::uint64_t
            {
                std::uint64_t w = v & 0xffffffffULL;
                w = (w | (w << 16)) & 0x0000ffff0000ffffULL;
                w = (w | (w <<  8)) & 0x00ff00ff00ff00ffULL;
                w = (w | (w <<  4)) & 0x0f0f0f0f0f0f0f0fULL;
                w = (w | (w <<  2)) & 0x3333333333333333ULL;
                w = (w | (w <<  1)) & 0x5555555555555555ULL;
                return w;
            };
            return spread(x) | (spread(y) << 1);
        };

        const int total = tiles_x * tiles_y;
        tile_order.resize(total);
        for (int i = 0; i < total; ++i) tile_order[i] = i;
        std::sort(tile_order.begin(), tile_order.end(),
            [&](int a, int b)
            {
                const std::uint64_t ka = morton_key(
                    static_cast<std::uint32_t>(a % tiles_x),
                    static_cast<std::uint32_t>(a / tiles_x));
                const std::uint64_t kb = morton_key(
                    static_cast<std::uint32_t>(b % tiles_x),
                    static_cast<std::uint32_t>(b / tiles_x));
                return ka < kb;
            });
    }

    std::uint32_t ray_tracer::seed_for(std::uint32_t ep, int spp, int thread_id) noexcept
    {
        std::uint32_t k = ep * 2654435761u;
        k ^= static_cast<std::uint32_t>(spp) + 0x9e3779b9u;
        k ^= static_cast<std::uint32_t>(thread_id) * 1973u + 9277u;
        return k ? k : 1u;
    }

    void ray_tracer::dispatcher_loop()
    {
        while (!stop_flag.load())
        {
            if (pause_requested.load(std::memory_order_acquire) || stop_flag.load())
            {
                std::unique_lock<std::mutex> lk(dispatch_mtx);
                dispatcher_paused = true;
                dispatch_cv.notify_all();
                dispatch_cv.wait(lk, [&] {
                    return !pause_requested.load(std::memory_order_acquire) || stop_flag.load();
                });
                dispatcher_paused = false;
                if (stop_flag.load()) return;
                continue;
            }

            const int live_target = std::max(1, config().target_spp.load(std::memory_order_relaxed));
            if (render_film->SPP.load(std::memory_order_relaxed) >= live_target
                || config().pause_render.load(std::memory_order_relaxed))
            {
                std::unique_lock<std::mutex> lk(dispatch_mtx);
                dispatch_cv.wait_for(lk, std::chrono::milliseconds(50), [&] {
                    const int t = std::max(1, config().target_spp.load(std::memory_order_relaxed));
                    return stop_flag.load()
                        || pause_requested.load(std::memory_order_acquire)
                        || (render_film->SPP.load(std::memory_order_relaxed) < t
                            && !config().pause_render.load(std::memory_order_relaxed));
                });
                continue;
            }

            const int batch = std::max(1, config().samples_per_call.load(std::memory_order_relaxed));
            for (int si = 0; si < batch; ++si)
            {
                if (stop_flag.load() || pause_requested.load(std::memory_order_acquire)) break;
                if (config().pause_render.load(std::memory_order_relaxed)) break;
                if (render_film->SPP.load(std::memory_order_relaxed)
                    >= std::max(1, config().target_spp.load(std::memory_order_relaxed))) break;
                run_one_pass();
            }
        }
    }

    void ray_tracer::run_one_pass()
    {
        render_film->increment_spp();
        const std::uint32_t ep  = epoch.load(std::memory_order_relaxed);
        const int           spp = render_film->SPP.load(std::memory_order_relaxed);

        for (int i = 0; i < num_procs; ++i)
        {
            if (auto* mt = dynamic_cast<mt_random*>(samplers[i].get()))
            {
                mt->generator.seed(seed_for(ep, spp, i));
            }
            else if (auto* ps = dynamic_cast<sampling::pixel_sampler*>(samplers[i].get()))
            {
                ps->reset_with_seed(seed_for(ep, spp, i));
            }
        }

        const int mode = config().render_mode.load(std::memory_order_relaxed);
        if (static_cast<render_mode>(mode) == render_mode::photon_map)
        {
            const bool need_shoot =
                (pm_global == nullptr || pm_global->size() == 0);

            if (need_shoot && photon_shoot != nullptr)
            {
                photon_shoot->p_global = static_cast<std::size_t>(std::max(0,
                    config().pm_p_global.load(std::memory_order_relaxed)));
                photon_shoot->p_caustic = static_cast<std::size_t>(std::max(0,
                    config().pm_p_caustic.load(std::memory_order_relaxed)));

                pool_phase_mode.store(static_cast<int>(pool_phase::photon_shoot_g),
                                      std::memory_order_release);
                run_pass_on_pool();

                if (pm_global) pm_global->merge_pending();

                pool_phase_mode.store(static_cast<int>(pool_phase::photon_shoot_c),
                                      std::memory_order_release);
                run_pass_on_pool();
                if (pm_caustic) pm_caustic->merge_pending();
            }

            if (photon_gather != nullptr)
            {
                photon_gather->k_global          = std::max(1,
                    config().pm_k_global .load(std::memory_order_relaxed));
                photon_gather->k_caustic         = std::max(1,
                    config().pm_k_caustic.load(std::memory_order_relaxed));
                photon_gather->r_max_global      =
                    config().pm_r_max_global .load(std::memory_order_relaxed);
                photon_gather->r_max_caustic     =
                    config().pm_r_max_caustic.load(std::memory_order_relaxed);
                photon_gather->use_final_gather  =
                    config().pm_use_final_gather .load(std::memory_order_relaxed);
                photon_gather->final_gather_rays = std::max(0,
                    config().pm_final_gather_rays.load(std::memory_order_relaxed));
            }

            refresh_adaptive_budget();
            pool_phase_mode.store(static_cast<int>(pool_phase::tile_pm),
                                  std::memory_order_release);
            run_pass_on_pool();
        }
        else if (static_cast<render_mode>(mode) == render_mode::vpl)
        {
            if (ir != nullptr)
            {
                std::lock_guard<std::mutex> lk(ir->vpls_mtx);
                ir->vpls.clear();
                for (auto& v : ir->pending_vpls) v.clear();
                ir->num_vpls = static_cast<std::size_t>(std::max(0,
                    config().ir_num_vpls.load(std::memory_order_relaxed)));
            }
            pool_phase_mode.store(static_cast<int>(pool_phase::vpl_shoot),
                                  std::memory_order_release);
            run_pass_on_pool();
            if (ir != nullptr) ir->merge_pending();

            refresh_adaptive_budget();
            pool_phase_mode.store(static_cast<int>(pool_phase::tile_vpl),
                                  std::memory_order_release);
            run_pass_on_pool();
        }
        else
        {
            refresh_adaptive_budget();
            pool_phase_mode.store(static_cast<int>(pool_phase::tile_render),
                                  std::memory_order_release);
            run_pass_on_pool();
        }
    }

    void ray_tracer::run_pass_on_pool()
    {
        pool_next_tile.store(0, std::memory_order_release);
        pool_done_counter.store(0, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lk(pool_mtx);
            pool_pass_epoch = epoch.load(std::memory_order_relaxed);
            pool_work_gen.fetch_add(1, std::memory_order_acq_rel);
        }
        pool_cv_start.notify_all();

        std::unique_lock<std::mutex> lk(pool_mtx);
        pool_cv_done.wait(lk, [&] {
            return pool_done_counter.load(std::memory_order_acquire) >= num_procs;
        });
    }

    void ray_tracer::shutdown_pool()
    {
        if (pool_threads.empty()) return;
        {
            std::lock_guard<std::mutex> lk(pool_mtx);
            pool_shutdown.store(true, std::memory_order_release);
        }
        pool_cv_start.notify_all();
        for (auto& th : pool_threads) if (th.joinable()) th.join();
        pool_threads.clear();
    }

    color ray_tracer::sample_for_mode(geometry::ray& r, sampler* s, int mode)
    {
        switch (static_cast<render_mode>(mode))
        {
        case render_mode::direct:  return direct_only(r, s);
        case render_mode::albedo:  return albedo(r);
        case render_mode::normals: return view_normals(r);
        case render_mode::path_trace:
        default:                   return path_trace(r, s);
        }
    }

    void ray_tracer::shade_tile(unsigned int x0, unsigned int y0,
                                unsigned int x1, unsigned int y1,
                                sampler* s, std::uint32_t ep)
    {
        const int   mode = config().render_mode.load(std::memory_order_relaxed);
        const bool  use_fis = config().use_filter_importance_sampling
                                  .load(std::memory_order_relaxed);

        const bool adaptive_active =
            config().use_adaptive_sampling.load(std::memory_order_relaxed)
            && adaptive != nullptr
            && render_film->SPP.load(std::memory_order_relaxed)
                   >= std::max(0, config().adaptive_warmup_spp
                                       .load(std::memory_order_relaxed));
        const int max_per_pixel = std::max(1,
            config().adaptive_max_per_pixel.load(std::memory_order_relaxed));

        tonemap_params tm;
        tm.exposure    = config().exposure  .load(std::memory_order_relaxed);
        tm.gamma       = std::max(0.01f, config().gamma.load(std::memory_order_relaxed));
        tm.contrast    = config().contrast  .load(std::memory_order_relaxed);
        tm.saturation  = config().saturation.load(std::memory_order_relaxed);

        const int filter_radius = render_film->filter->size();
        const int pad = filter_radius;
        const int buf_x0 = static_cast<int>(x0) - pad;
        const int buf_y0 = static_cast<int>(y0) - pad;
        const int buf_w  = static_cast<int>(x1 - x0) + 2 * pad;
        const int buf_h  = static_cast<int>(y1 - y0) + 2 * pad;

        // std::vector<color> scratch(static_cast<size_t>(buf_w) * buf_h);
        thread_local std::vector<color> scratch_tls;
        const size_t needed = static_cast<size_t>(buf_w) * buf_h;

        if (scratch_tls.size() < needed)
            scratch_tls.resize(needed);

        std::fill_n(scratch_tls.begin(), needed, color(0.0f, 0.0f, 0.0f));
        color* const scratch = scratch_tls.data();

        for (unsigned int y = y0; y < y1; ++y)
        {
            for (unsigned int x = x0; x < x1; ++x)
            {
                const int   N     = adaptive_samples_for(static_cast<int>(x),
                                                         static_cast<int>(y),
                                                         adaptive_active,
                                                         max_per_pixel);
                const float inv_N = 1.0f / static_cast<float>(N);
                for (int si = 0; si < N; ++si)
                {
                    const float px = static_cast<float>(x) + s->next();
                    const float py = static_cast<float>(y) + s->next();

                    geometry::ray r = target_scene->cam.generate_ray_thin_lens(px, py, s);
                    const color col = sample_for_mode(r, s, mode) * inv_N;
                    if (use_fis)
                    {
                        render_film->splat_importance(
                            scratch, buf_w, buf_h, buf_x0, buf_y0,
                            px, py, col, s->next(), s->next());
                    }
                    else
                    {
                        render_film->splat_into(
                            scratch, buf_w, buf_h, buf_x0, buf_y0,
                            px, py, col);
                    }
                }
            }
        }

        if (epoch.load(std::memory_order_acquire) != ep) return;

        merge_scratch(scratch, buf_w, buf_h, buf_x0, buf_y0);

        for (unsigned int y = y0; y < y1; ++y)
        {
            for (unsigned int x = x0; x < x1; ++x)
            {
                unsigned char r8, g8, b8;
                render_film->tonemap(static_cast<int>(x), static_cast<int>(y),
                                     r8, g8, b8, tm);
                canvas->draw(x, y, r8, g8, b8);
            }
        }
    }

    void ray_tracer::merge_scratch(color* scratch,
                                   int buf_w, int buf_h,
                                   int buf_x0, int buf_y0)
    {
        const int fw = static_cast<int>(render_film->width);
        const int fh = static_cast<int>(render_film->height);

        for (int ly = 0; ly < buf_h; ++ly)
        {
            const int fy = buf_y0 + ly;
            if (fy < 0 || fy >= fh) continue;

            // std::lock_guard<std::mutex> lock(
            //     merge_stripes[static_cast<unsigned>(fy) &
            //         (renderer_config::merge_strips - 1)]);
            // for (int lx = 0; lx < buf_w; ++lx)
            // {
            //     const int fx = buf_x0 + lx;
            //     if (fx < 0 || fx >= fw) continue;
            //     const color& c = scratch[ly * buf_w + lx];
            //     if (c.red == 0.0f && c.green == 0.0f && c.blue == 0.0f) continue;
            //     render_film->film_buffer[fy * fw + fx] =
            //         render_film->film_buffer[fy * fw + fx] + c;
            // }

            const color* row = scratch + ly * buf_w;
            bool any = false;
            for (int lx = 0; lx < buf_w; ++lx)
            {
                if (row[lx].red != 0.0f || row[lx].green != 0.0f || row[lx].blue != 0.0f)
                {
                    any = true;
                    break;
                }
            }
            if (!any) continue;

            std::lock_guard<std::mutex> lock(
                merge_stripes[static_cast<unsigned>(fy) &
                    (renderer_config::merge_strips - 1)]);
            for (int lx = 0; lx < buf_w; ++lx)
            {
                const int fx = buf_x0 + lx;
                if (fx < 0 || fx >= fw) continue;
                const color& c = row[lx];
                if (c.red == 0.0f && c.green == 0.0f && c.blue == 0.0f) continue;
                render_film->film_buffer[fy * fw + fx] =
                    render_film->film_buffer[fy * fw + fx] + c;
            }
        }
    }

    void ray_tracer::shade_tile_light(unsigned int x0, unsigned int y0,
                                      unsigned int x1, unsigned int y1,
                                      sampler* s, std::uint32_t ep)
    {
        if (lt == nullptr) return;

        const unsigned int n_paths = (x1 - x0) * (y1 - y0);
        for (unsigned int i = 0; i < n_paths; ++i)
        {
            if (epoch.load(std::memory_order_acquire) != ep) return;
            lt->light_trace(s);
        }
    }

    void ray_tracer::tonemap_tile(unsigned int x0, unsigned int y0,
                                  unsigned int x1, unsigned int y1)
    {
        if (canvas == nullptr || render_film == nullptr) return;
        if (canvas_locked_.load(std::memory_order_acquire)) return;

        tonemap_params tm;
        tm.exposure    = config().exposure  .load(std::memory_order_relaxed);
        tm.gamma       = std::max(0.01f, config().gamma.load(std::memory_order_relaxed));
        tm.contrast    = config().contrast  .load(std::memory_order_relaxed);
        tm.saturation  = config().saturation.load(std::memory_order_relaxed);

        for (unsigned int y = y0; y < y1; ++y)
        {
            for (unsigned int x = x0; x < x1; ++x)
            {
                unsigned char r8, g8, b8;
                render_film->tonemap(static_cast<int>(x), static_cast<int>(y),
                                     r8, g8, b8, tm);
                canvas->draw(x, y, r8, g8, b8);
            }
        }
    }

    void ray_tracer::shoot_photons_chunk(unsigned int n_paths,
                                         sampler* s, bool for_caustic,
                                         std::uint32_t ep,
                                         int worker_id)
    {
        if (photon_shoot == nullptr) return;
        for (unsigned int i = 0; i < n_paths; ++i)
        {
            if (epoch.load(std::memory_order_acquire) != ep) return;
            photon_shoot->shoot_one(s, for_caustic, worker_id);
        }
    }

    void ray_tracer::shade_tile_pm(unsigned int x0, unsigned int y0,
                                   unsigned int x1, unsigned int y1,
                                   sampler* s, std::uint32_t ep)
    {
        if (photon_gather == nullptr)
        {
            shade_tile(x0, y0, x1, y1, s, ep);
            return;
        }

        const bool use_fis = config().use_filter_importance_sampling
                                 .load(std::memory_order_relaxed);

        const bool adaptive_active =
            config().use_adaptive_sampling.load(std::memory_order_relaxed)
            && adaptive != nullptr
            && render_film->SPP.load(std::memory_order_relaxed)
                   >= std::max(0, config().adaptive_warmup_spp
                                       .load(std::memory_order_relaxed));
        const int max_per_pixel = std::max(1,
            config().adaptive_max_per_pixel.load(std::memory_order_relaxed));

        tonemap_params tm;
        tm.exposure    = config().exposure  .load(std::memory_order_relaxed);
        tm.gamma       = std::max(0.01f, config().gamma.load(std::memory_order_relaxed));
        tm.contrast    = config().contrast  .load(std::memory_order_relaxed);
        tm.saturation  = config().saturation.load(std::memory_order_relaxed);

        const int filter_radius = render_film->filter->size();
        const int pad = filter_radius;
        const int buf_x0 = static_cast<int>(x0) - pad;
        const int buf_y0 = static_cast<int>(y0) - pad;
        const int buf_w  = static_cast<int>(x1 - x0) + 2 * pad;
        const int buf_h  = static_cast<int>(y1 - y0) + 2 * pad;

        // std::vector<color> scratch(static_cast<size_t>(buf_w) * buf_h);

        thread_local std::vector<color> scratch_tls;
        const size_t needed = static_cast<size_t>(buf_w) * buf_h;

        if (scratch_tls.size() < needed)
            scratch_tls.resize(needed);

        std::fill_n(scratch_tls.begin(), needed, color(0.0f, 0.0f, 0.0f));
        color* const scratch = scratch_tls.data();

        const photon_gatherer::direct_cb direct_cb =
            [this](const shading_data& sd_in, sampler* ss)
            { return this->compute_direct(sd_in, ss); };

        for (unsigned int y = y0; y < y1; ++y)
        {
            for (unsigned int x = x0; x < x1; ++x)
            {
                const int   N     = adaptive_samples_for(static_cast<int>(x),
                                                         static_cast<int>(y),
                                                         adaptive_active,
                                                         max_per_pixel);
                const float inv_N = 1.0f / static_cast<float>(N);
                for (int si = 0; si < N; ++si)
                {
                    const float px = static_cast<float>(x) + s->next();
                    const float py = static_cast<float>(y) + s->next();

                    geometry::ray r = target_scene->cam.generate_ray_thin_lens(px, py, s);
                    const color col = photon_gather->shade_eye(r, s, direct_cb) * inv_N;
                    if (use_fis)
                    {
                        render_film->splat_importance(
                            scratch, buf_w, buf_h, buf_x0, buf_y0,
                            px, py, col, s->next(), s->next());
                    }
                    else
                    {
                        render_film->splat_into(
                            scratch, buf_w, buf_h, buf_x0, buf_y0,
                            px, py, col);
                    }
                }
            }
        }

        if (epoch.load(std::memory_order_acquire) != ep) return;

        merge_scratch(scratch, buf_w, buf_h, buf_x0, buf_y0);

        for (unsigned int y = y0; y < y1; ++y)
        {
            for (unsigned int x = x0; x < x1; ++x)
            {
                unsigned char r8, g8, b8;
                render_film->tonemap(static_cast<int>(x), static_cast<int>(y),
                                     r8, g8, b8, tm);
                canvas->draw(x, y, r8, g8, b8);
            }
        }
    }

    void ray_tracer::shoot_vpls_chunk(unsigned int n_paths,
                                      sampler* s, std::uint32_t ep,
                                      int worker_id)
    {
        if (ir == nullptr) return;
        for (unsigned int i = 0; i < n_paths; ++i)
        {
            if (epoch.load(std::memory_order_acquire) != ep) return;
            ir->shoot_one_path(s, worker_id);
        }
    }

    void ray_tracer::shade_tile_vpl(const unsigned int x0, const unsigned int y0,
                                    const unsigned int x1, const unsigned int y1,
                                    sampler* s, const std::uint32_t ep)
    {
        if (ir == nullptr) { shade_tile(x0, y0, x1, y1, s, ep); return; }

        const bool use_fis = config().use_filter_importance_sampling
                                 .load(std::memory_order_relaxed);

        const bool adaptive_active =
            config().use_adaptive_sampling.load(std::memory_order_relaxed)
            && adaptive != nullptr
            && render_film->SPP.load(std::memory_order_relaxed)
                   >= std::max(0, config().adaptive_warmup_spp
                                       .load(std::memory_order_relaxed));
        const int max_per_pixel = std::max(1,
            config().adaptive_max_per_pixel.load(std::memory_order_relaxed));

        tonemap_params tm;
        tm.exposure    = config().exposure  .load(std::memory_order_relaxed);
        tm.gamma       = std::max(0.01f, config().gamma.load(std::memory_order_relaxed));
        tm.contrast    = config().contrast  .load(std::memory_order_relaxed);
        tm.saturation  = config().saturation.load(std::memory_order_relaxed);

        const int filter_radius = render_film->filter->size();
        const int pad = filter_radius;
        const int buf_x0 = static_cast<int>(x0) - pad;
        const int buf_y0 = static_cast<int>(y0) - pad;
        const int buf_w  = static_cast<int>(x1 - x0) + 2 * pad;
        const int buf_h  = static_cast<int>(y1 - y0) + 2 * pad;

        // std::vector<color> scratch(static_cast<size_t>(buf_w) * buf_h);
        thread_local std::vector<color> scratch_tls;
        const size_t needed = static_cast<size_t>(buf_w) * buf_h;

        if (scratch_tls.size() < needed)
            scratch_tls.resize(needed);

        std::fill_n(scratch_tls.begin(), needed, color(0.0f, 0.0f, 0.0f));
        color* const scratch = scratch_tls.data();

        for (unsigned int y = y0; y < y1; ++y)
        {
            for (unsigned int x = x0; x < x1; ++x)
            {
                const int   N     = adaptive_samples_for(static_cast<int>(x),
                                                         static_cast<int>(y),
                                                         adaptive_active,
                                                         max_per_pixel);
                const float inv_N = 1.0f / static_cast<float>(N);
                for (int si = 0; si < N; ++si)
                {
                    const float px = static_cast<float>(x) + s->next();
                    const float py = static_cast<float>(y) + s->next();

                    geometry::ray r = target_scene->cam.generate_ray_thin_lens(px, py, s);
                    const color col = ir->shade_eye(r, s) * inv_N;
                    if (use_fis)
                    {
                        render_film->splat_importance(
                            scratch, buf_w, buf_h, buf_x0, buf_y0,
                            px, py, col, s->next(), s->next());
                    }
                    else
                    {
                        render_film->splat_into(
                            scratch, buf_w, buf_h, buf_x0, buf_y0,
                            px, py, col);
                    }
                }
            }
        }

        if (epoch.load(std::memory_order_acquire) != ep) return;

        merge_scratch(scratch, buf_w, buf_h, buf_x0, buf_y0);

        for (unsigned int y = y0; y < y1; ++y)
        {
            for (unsigned int x = x0; x < x1; ++x)
            {
                unsigned char r8, g8, b8;
                render_film->tonemap(static_cast<int>(x), static_cast<int>(y),
                                     r8, g8, b8, tm);
                canvas->draw(x, y, r8, g8, b8);
            }
        }
    }
} // namespace fox_tracer
