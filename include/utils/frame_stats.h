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
#ifndef RAYTRACER_WITH_AI_FRAME_STATS_H
#define RAYTRACER_WITH_AI_FRAME_STATS_H

#include <chrono>
#include <cstdint>

namespace fox_tracer::utility
{
    class frame_stats
    {
    public:
        void reset (int initial_spp) noexcept;
        void update(int current_spp, int target_spp) noexcept;

        [[nodiscard]] float fps_ema     () const noexcept { return fps_ema_; }
        [[nodiscard]] float spp_per_sec () const noexcept { return spp_per_sec_; }
        [[nodiscard]] float eta_sec_ema () const noexcept { return eta_sec_ema_; }
        [[nodiscard]] bool  eta_valid   () const noexcept { return eta_valid_; }

    private:
        void rebaseline_rate(std::chrono::steady_clock::time_point now,
                             int current_spp) noexcept;

        std::chrono::steady_clock::time_point last_frame_tp_{};
        std::chrono::steady_clock::time_point stats_last_tp_{};
        int   stats_last_spp_   {0};
        float fps_ema_          {0.0f};
        float spp_per_sec_      {0.0f};
        float eta_sec_ema_      {0.0f};
        bool  eta_valid_        {false};

        std::chrono::steady_clock::time_point eta_baseline_tp_{};
        int eta_baseline_spp_{0};

        std::uint32_t cached_reset_gen_{0};
        bool          cached_paused_   {false};
    };
} // namespace fox_tracer
#endif //RAYTRACER_WITH_AI_FRAME_STATS_H
