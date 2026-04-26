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
#include "utils/frame_stats.h"

#include "config.h"

#include <algorithm>

namespace fox_tracer::utility
{
    void frame_stats::reset(int initial_spp) noexcept
    {
        last_frame_tp_  = std::chrono::steady_clock::now();
        stats_last_tp_  = last_frame_tp_;
        stats_last_spp_ = initial_spp;
        fps_ema_        = 0.0f;
        spp_per_sec_    = 0.0f;
        eta_sec_ema_    = 0.0f;
        eta_valid_      = false;
        eta_baseline_tp_  = last_frame_tp_;
        eta_baseline_spp_ = initial_spp;
        cached_reset_gen_ =
            config().reset_generation.load(std::memory_order_relaxed);
        cached_paused_  =
            config().pause_render.load(std::memory_order_relaxed);
    }

    void frame_stats::rebaseline_rate(
        std::chrono::steady_clock::time_point now,
        int current_spp) noexcept
    {
        stats_last_tp_  = now;
        stats_last_spp_ = current_spp;
        spp_per_sec_    = 0.0f;
        eta_sec_ema_    = 0.0f;
        eta_valid_      = false;
        eta_baseline_tp_  = now;
        eta_baseline_spp_ = current_spp;
    }

    void frame_stats::update(int current_spp, int target_spp) noexcept
    {
        const auto now = std::chrono::steady_clock::now();
        const float frame_sec =
            std::chrono::duration<float>(now - last_frame_tp_).count();
        last_frame_tp_ = now;
        if (frame_sec > 0.0f)
        {
            const float inst = 1.0f / frame_sec;
            fps_ema_ = (fps_ema_ <= 0.0f) ? inst : (fps_ema_ * 0.9f + inst * 0.1f);
        }

        const std::uint32_t cur_reset_gen =
            config().reset_generation.load(std::memory_order_relaxed);
        const bool paused_now =
            config().pause_render.load(std::memory_order_relaxed);

        const bool spp_regressed     = (current_spp < stats_last_spp_);
        const bool reset_bumped      = (cur_reset_gen != cached_reset_gen_);
        const bool resumed           = (cached_paused_ && !paused_now);

        if (spp_regressed || reset_bumped || resumed)
        {
            rebaseline_rate(now, current_spp);
        }
        cached_reset_gen_ = cur_reset_gen;
        cached_paused_    = paused_now;

        if (paused_now)
        {
            stats_last_tp_  = now;
            stats_last_spp_ = current_spp;
        }

        const float window_sec =
            std::chrono::duration<float>(now - stats_last_tp_).count();
        if (window_sec >= 0.25f)
        {
            const int delta = current_spp - stats_last_spp_;
            if (delta > 0 && window_sec > 0.0f)
            {
                const float inst = static_cast<float>(delta) / window_sec;
                spp_per_sec_ = (spp_per_sec_ <= 0.0f)
                    ? inst
                    : (spp_per_sec_ * 0.85f + inst * 0.15f);
                stats_last_tp_  = now;
                stats_last_spp_ = current_spp;
            }
            else if (window_sec > 1.0f)
            {
                spp_per_sec_ *= 0.5f;
                if (spp_per_sec_ < 1e-3f) spp_per_sec_ = 0.0f;
                stats_last_tp_  = now;
                stats_last_spp_ = current_spp;
            }
        }

        const int target = std::max(1, target_spp);
        const int rem    = std::max(0, target - current_spp);

        const float baseline_sec =
            std::chrono::duration<float>(now - eta_baseline_tp_).count();
        const int   baseline_delta = current_spp - eta_baseline_spp_;
        constexpr float eta_cap_sec = 99.0f * 3600.0f;

        if (rem == 0)
        {
            eta_sec_ema_ = 0.0f;
            eta_valid_   = true;
        }
        else if (baseline_delta > 0 && baseline_sec >= 0.5f)
        {
            const float sec_per_spp =
                baseline_sec / static_cast<float>(baseline_delta);
            float eta = static_cast<float>(rem) * sec_per_spp;
            if (eta > eta_cap_sec) eta = eta_cap_sec;

            eta_sec_ema_ = (eta_sec_ema_ <= 0.0f)
                ? eta
                : (eta_sec_ema_ * 0.5f + eta * 0.5f);
            eta_valid_ = true;
        }
        else
        {
            if (eta_sec_ema_ <= 0.0f) eta_valid_ = false;
        }
    }
} // namespace fox_tracer
