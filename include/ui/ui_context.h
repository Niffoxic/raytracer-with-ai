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
#ifndef RAYTRACER_WITH_AI_UI_CONTEXT_H
#define RAYTRACER_WITH_AI_UI_CONTEXT_H


#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace GamesEngineeringBase
{
    class Window;
}

namespace fox_tracer
{
    namespace render
    {
        class ray_tracer;
    }
    namespace scene
    {
        class scene_editor;
    }
    namespace utility
    {
        class effect_runner;
    }

    namespace ui
    {
        class image_gallery;

        struct ui_context
        {
            render::ray_tracer*           rt        {nullptr};
            scene::scene_editor*          editor    {nullptr};
            image_gallery*                gallery   {nullptr};
            utility::effect_runner*       fx_runner {nullptr};
            GamesEngineeringBase::Window* window    {nullptr};

            float fps_ema     {0.0f};
            float spp_per_sec {0.0f};
            float eta_sec_ema {0.0f};
            bool  eta_valid   {false};

            const std::string*       fx_phase_label         {nullptr};
            const float*             fx_phase_progress      {nullptr};
            const std::string*       fx_device              {nullptr};
            const std::atomic<bool>* bootstrap_busy         {nullptr};
            std::mutex*              bootstrap_mtx          {nullptr};
            const std::string*       bootstrap_last_summary {nullptr};

            std::function<void()>                   trigger_denoise;
            std::function<void(const std::string&)> launch_effect;
            std::function<void()>                   rescan_effects;
            std::function<void()>                   restart_worker;
            std::function<void()>                   create_venv;

            std::function<void(std::string)>        install_ai_deps_now;
            std::function<std::string()>            resolve_venv_python;
            std::function<bool()>                   cuda_available;
            std::function<void()>                   refresh_cuda_detection;
        };
    } // namespace ui
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_UI_CONTEXT_H
