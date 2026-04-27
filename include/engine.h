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
#ifndef RAYTRACER_WITH_AI_ENGINE_H
#define RAYTRACER_WITH_AI_ENGINE_H

#include "utils/effect_pipeline.h"
#include "utils/frame_stats.h"
#include "utils/input_controller.h"
#include "utils/python_env.h"
#include "render/renderer.h"

#include "scene/scene_host.h"
#include "ui/ui_panel.h"

#include <memory>
#include <string>

namespace GamesEngineeringBase { class Window; }

namespace fox_tracer
{
    struct engine_config
    {
        std::string scene_name;
        std::string assets_root{"assets"};
        int         width  { 1280 };
        int         height { 720 };
        bool        show_settings_panel{true};
    };

    class engine
    {
    public:
         engine() noexcept;
        ~engine();

        engine(const engine&)            = delete;
        engine& operator=(const engine&) = delete;
        engine(engine&&)                 = delete;
        engine& operator=(engine&&)      = delete;

        int run(int argc, char** argv);
        int run(const engine_config& cfg);

    private:
        bool init(const engine_config& cfg);
        void main_loop();
        void shutdown ();

        void build_ui_context(ui::ui_context& out);
        void trigger_denoise ();

        static engine_config parse_args(int argc, char** argv);

        std::unique_ptr<GamesEngineeringBase::Window> window_;
        render::ray_tracer          rt_;
        scene::scene_host           scene_host_;
        utility::effect_pipeline    fx_;
        utility::python_env         python_env_;
        utility::frame_stats        stats_;
        utility::input_controller   input_;
        ui::ui_panel                ui_;

        engine_config cfg_;
    };
} // namespace fox_tracer


#endif //RAYTRACER_WITH_AI_ENGINE_H
