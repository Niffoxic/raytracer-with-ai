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
#include "utils/effect_pipeline.h"
#include "utils/logger.h"
#include "utils/paths.h"
#include "framework/base.h"

#include <cstddef>

namespace fox_tracer::utility
{
    void effect_pipeline::start()
    {
        if (started_) return;

        const std::string script = paths::resolve("assets/scripts/host.py");
        if (!paths::exists(script))
        {
            LOG_WARN("py_effect") << "host.py not found at " << script
                                  << " - cant fetch effect";
            started_ = true;
            return;
        }

        started_ = true;
        if (!runner_.start(script))
        {
            LOG_WARN("py_effect") << "failed to start python worker: "
                                  << runner_.last_error();
            return;
        }

        runner_.request_list();
    }

    void effect_pipeline::shutdown()
    {
        runner_.shutdown();
    }

    void effect_pipeline::reset_pending_state()
    {
        phase_label_.clear();
        phase_progress_ = 0.0f;
        device_.clear();
        pending_entry_id_ = 0;
        pending_input_rgb_.clear();
        pending_input_rgb_.shrink_to_fit();
        pending_infer_after_load_ = false;
    }

    void effect_pipeline::rescan()
    {
        if (!runner_.alive()) return;
        runner_.request_list();
    }

    void effect_pipeline::restart()
    {
        reset_pending_state();

        if (runner_.restart())
        {
            started_ = true;
            LOG_INFO("py_effect") << "worker restarted (python="
                                  << runner_.picked_python() << ")";
            runner_.request_list();
        }
        else
        {
            LOG_WARN("py_effect") << "restart failed trying cold init";
            started_ = false;
            start();
        }
    }

    void effect_pipeline::launch(
        const std::string& effect_name,
        const GamesEngineeringBase::Window* window,
        bool& show_images_panel)
    {
        if (!runner_.alive() || window == nullptr) return;

        const int w = static_cast<int>(window->getWidth());
        const int h = static_cast<int>(window->getHeight());
        if (w <= 0 || h <= 0) return;

        pending_input_rgb_.assign(
            window->getBackBuffer(),
            window->getBackBuffer() + static_cast<std::size_t>(w) * h * 3);
        pending_effect_name_      = effect_name;
        pending_w_                = w;
        pending_h_                = h;
        pending_infer_after_load_ = true;

        pending_entry_id_ = gallery_.begin_effect(
            effect_name, pending_input_rgb_.data(), w, h);

        if (!show_images_panel) show_images_panel = true;

        phase_label_.clear();
        phase_progress_ = 0.0f;
        if (effect_name != runner_.active_effect_name())
            device_.clear();

        if (!runner_.request_load(effect_name))
        {
            gallery_.fail_effect(pending_entry_id_, runner_.last_error());
            pending_entry_id_         = 0;
            pending_infer_after_load_ = false;
        }
    }

    void effect_pipeline::poll()
    {
        if (!runner_.alive() && !started_) return;

        std::vector<effect_runner::event> events;
        runner_.poll(events);

        for (auto& ev : events)
        {
            using k = effect_runner::event_kind;
            switch (ev.kind)
            {
            case k::hello:
                LOG_INFO("py_effect") << "worker ready";
                break;
            case k::list:
                LOG_INFO("py_effect") << "discovered "
                                      << ev.effects.size() << " effect(s)";
                break;
            case k::loaded:
                if (pending_infer_after_load_ && pending_entry_id_ != 0 &&
                    !pending_input_rgb_.empty())
                {
                    pending_infer_after_load_ = false;
                    if (!runner_.request_infer(pending_input_rgb_.data(),
                                               pending_w_, pending_h_))
                    {
                        gallery_.fail_effect(pending_entry_id_,
                                             runner_.last_error());
                        pending_entry_id_ = 0;
                    }
                    pending_input_rgb_.clear();
                    pending_input_rgb_.shrink_to_fit();
                }
                break;
            case k::load_error:
                if (pending_entry_id_ != 0)
                {
                    gallery_.fail_effect(pending_entry_id_,
                                         ev.message.empty() ? "load failed"
                                                            : ev.message);
                    pending_entry_id_ = 0;
                }
                pending_infer_after_load_ = false;
                pending_input_rgb_.clear();
                break;
            case k::partial:
                if (pending_entry_id_ != 0 && !ev.bytes.empty())
                {
                    gallery_.update_effect_partial(
                        pending_entry_id_,
                        ev.bytes.data(), ev.bytes.size(),
                        ev.progress);
                }
                break;
            case k::done:
                if (pending_entry_id_ != 0 && !ev.bytes.empty())
                {
                    gallery_.finalize_effect(pending_entry_id_,
                                             ev.bytes.data(),
                                             ev.bytes.size());
                }
                pending_entry_id_ = 0;
                break;
            case k::cancelled:
                if (pending_entry_id_ != 0)
                {
                    gallery_.cancel_effect(pending_entry_id_);
                }
                pending_entry_id_         = 0;
                pending_infer_after_load_ = false;
                pending_input_rgb_.clear();
                break;
            case k::error:
                if (pending_entry_id_ != 0)
                {
                    gallery_.fail_effect(pending_entry_id_,
                                         ev.message.empty() ? "unknown error"
                                                            : ev.message);
                    pending_entry_id_ = 0;
                }
                pending_infer_after_load_ = false;
                pending_input_rgb_.clear();
                LOG_WARN("py_effect") << "error: " << ev.message;
                break;
            case k::bye:
                LOG_INFO("py_effect") << "worker exited";
                phase_label_.clear();
                phase_progress_ = 0.0f;
                device_.clear();
                break;
            case k::status:
                phase_label_    = ev.message;
                phase_progress_ = ev.progress;
                {
                    constexpr char tag[] = "device=";
                    const auto p = ev.message.find(tag);
                    if (p != std::string::npos)
                    {
                        std::size_t s = p + sizeof(tag) - 1;
                        std::size_t e = s;
                        while (e < ev.message.size() &&
                               ev.message[e] != ' ' &&
                               ev.message[e] != ',' &&
                               ev.message[e] != ')')
                            ++e;
                        if (e > s) device_ = ev.message.substr(s, e - s);
                    }
                }
                break;
            case k::unloaded:
                phase_label_.clear();
                phase_progress_ = 0.0f;
                device_.clear();
                pending_infer_after_load_ = false;
                pending_input_rgb_.clear();
                pending_entry_id_ = 0;
                LOG_INFO("py_effect") << "effect unloaded";
                break;
            }
        }
    }
} // namespace fox_tracer
