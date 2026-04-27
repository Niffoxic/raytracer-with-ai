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
#include "engine.h"

#include "config.h"
#include "utils/logger.h"
#include "ui/ui_context.h"
#include "framework/base.h"

#include <windows.h>
#include <chrono>
#include <thread>
#include <utility>

namespace fox_tracer
{
    engine::engine() noexcept = default;

    engine::~engine()
    {
        shutdown();
    }

    engine_config engine::parse_args(int argc, char** argv)
    {
        engine_config cfg;
        cfg.scene_name = "assets/scene/cornell-box";

        if (argc >= 2) cfg.scene_name = argv[1];
        if (argc >= 4)
        {
            try
            {
                cfg.width  = std::stoi(argv[2]);
                cfg.height = std::stoi(argv[3]);
            }
            catch (...)
            {
                LOG_WARN("engine") << "using defaults";
            }
        }
        return cfg;
    }

    int engine::run(int argc, char** argv)
    {
        return run(parse_args(argc, argv));
    }

    int engine::run(const engine_config& cfg)
    {
        if (!init(cfg)) return 1;
        main_loop();
        return 0;
    }

    bool engine::init(const engine_config& cfg)
    {
        cfg_ = cfg;
        ui_.show_settings_panel = cfg.show_settings_panel;

        window_ = std::make_unique<GamesEngineeringBase::Window>();
        window_->create(cfg.width, cfg.height, "fox_tracer");

        if (!scene_host_.init(cfg.scene_name, cfg.assets_root,
                              cfg.width, cfg.height))
        {
            return false;
        }

        rt_.init(scene_host_.current(), window_.get());
        rt_.start();

        if (config().num_threads.load(std::memory_order_relaxed) <= 0)
        {
            const int hw = std::max(1,
                static_cast<int>(std::thread::hardware_concurrency()));
            config().num_threads.store(hw, std::memory_order_relaxed);
        }

        stats_.reset(rt_.get_spp());

        fx_.start();

        return true;
    }

    void engine::main_loop()
    {
        while (!window_->keyPressed(VK_ESCAPE))
        {
            window_->checkInput();

            window_->beginImGuiFrame();

            input_.process(window_.get());
            scene_host_.check_pending_reset(rt_);
            scene_host_.check_pending_editor(rt_);
            rt_.poll_denoise();

            if (python_env_.consume_pending_restart())
            {
                LOG_INFO("py_effect")
                    << "bootstrap finished restarting worker to pick "
                       "up the new environment";
                fx_.restart();
            }

            fx_.poll();
            fx_.gallery().refresh_canvas_if_viewing(window_.get());

            stats_.update(rt_.get_spp(),
                          config().target_spp.load(std::memory_order_relaxed));

            ui::ui_context ctx;
            build_ui_context(ctx);
            ui_.draw(ctx);

            window_->present();

            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }

    void engine::build_ui_context(ui::ui_context& out)
    {
        out.rt        = &rt_;
        out.editor    = &scene_host_.editor();
        out.gallery   = &fx_.gallery();
        out.fx_runner = &fx_.runner();
        out.window    = window_.get();

        out.fps_ema     = stats_.fps_ema();
        out.spp_per_sec = stats_.spp_per_sec();
        out.eta_sec_ema = stats_.eta_sec_ema();
        out.eta_valid   = stats_.eta_valid();

        out.fx_phase_label         = fx_.phase_label_ptr();
        out.fx_phase_progress      = fx_.phase_progress_ptr();
        out.fx_device              = fx_.device_ptr();
        out.bootstrap_busy         = python_env_.busy_flag();
        out.bootstrap_mtx          = &python_env_.last_summary_mutex();
        out.bootstrap_last_summary = &python_env_.last_summary_unlocked();

        out.trigger_denoise        = [this]               { trigger_denoise(); };
        out.launch_effect          = [this](const std::string& n)
        {
            fx_.launch(n, window_.get(), ui_.show_images_panel);
        };
        out.rescan_effects         = [this]               { fx_.rescan(); };
        out.restart_worker         = [this]               { fx_.restart(); };
        out.create_venv            = [this]               { python_env_.create_venv(); };
        out.install_ai_deps_now    = [this](std::string u)
        {
            python_env_.install_ai_deps_now(std::move(u),
                                            fx_.runner().picked_python());
        };
        out.resolve_venv_python    = [this]               { return python_env_.resolve_venv_python(); };
        out.cuda_available         = [this]               { return python_env_.cuda_available(); };
        out.refresh_cuda_detection = [this]               { python_env_.refresh_cuda_detection(); };
    }

    void engine::trigger_denoise()
    {
        if (!rt_.denoiser_available())
        {
            LOG_WARN("engine") << "denoiser not available";
            return;
        }
        if (!rt_.denoise_async())
        {
            LOG_WARN("engine") << "denoise_async rejected (already running / 0 spp)";
        }
    }

    void engine::shutdown()
    {
        fx_.shutdown();
        rt_.stop();
        scene_host_.release();
        window_.reset();
    }
} // namespace fox_tracer
