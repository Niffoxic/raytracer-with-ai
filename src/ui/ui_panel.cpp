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
#include "ui/ui_panel.h"
#include "ui/ai_panel.h"
#include "ui/ui_denoise.h"
#include "ui/main_menu_bar.h"
#include "ui/ui_settings.h"
#include "ui/ui_context.h"
#include "ui/image_gallery.h"

#include "scene/scene_editor.h"

namespace fox_tracer::ui
{
    ui_panel::ui_panel()
    {
        ai_panel_ = std::make_unique<ai_panel>(&show_images_panel);
        menu_bar_ = std::make_unique<main_menu_bar>(&show_settings_panel,
                                                    &show_images_panel,
                                                    ai_panel_.get());
        settings_ = std::make_unique<settings_panel>(&show_settings_panel);
        toast_    = std::make_unique<denoise_component>();

        components_.push_back(menu_bar_.get());
        components_.push_back(settings_.get());
        components_.push_back(ai_panel_.get());
        components_.push_back(toast_.get());
    }

    ui_panel::~ui_panel() = default;

    void ui_panel::draw(ui_context& ctx)
    {
        if (!components_.empty()) components_.front()->draw(ctx);

        if (ctx.editor != nullptr) ctx.editor->draw_editor();

        for (std::size_t i = 1; i < components_.size(); ++i)
        {
            if (components_[i] == ai_panel_.get())
            {
                if (ctx.gallery && ctx.window)
                {
                    ctx.gallery->draw_ui(
                        show_images_panel, ctx.window, ctx.rt,
                        [this, &ctx]{ ai_panel_->render_status_header(ctx); });
                }
            }
            components_[i]->draw(ctx);
        }
    }
} // namespace fox_tracer::ui
