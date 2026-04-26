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
#ifndef RAYTRACER_WITH_AI_UI_PANEL_H
#define RAYTRACER_WITH_AI_UI_PANEL_H

#include "ui/ui_component.h"

#include <memory>
#include <vector>

namespace fox_tracer::ui
{
    class main_menu_bar;
    class settings_panel;
    class ai_panel;
    class denoise_component;

    class ui_panel
    {
    public:
        ui_panel();
        ~ui_panel();

        ui_panel(const ui_panel&)            = delete;
        ui_panel& operator=(const ui_panel&) = delete;

        void draw(ui_context& ctx);

        bool show_settings_panel{true};
        bool show_images_panel  {false};

    private:
        std::unique_ptr<ai_panel>           ai_panel_;
        std::unique_ptr<main_menu_bar>      menu_bar_;
        std::unique_ptr<settings_panel>     settings_;
        std::unique_ptr<denoise_component>  toast_;
        std::vector<interface_component*>   components_;
    };
} // namespace fox_tracer::ui

#endif //RAYTRACER_WITH_AI_UI_PANEL_H
