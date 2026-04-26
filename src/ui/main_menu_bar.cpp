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
#include "ui/main_menu_bar.h"
#include "ui/ai_panel.h"
#include "ui/ui_context.h"
#include "ui/image_gallery.h"

#include "config.h"
#include "render/denoiser.h"
#include "render/renderer.h"
#include "scene/scene_editor.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <thread>

namespace fox_tracer::ui
{
    main_menu_bar::main_menu_bar(bool*     show_settings_panel,
                                 bool*     show_images_panel,
                                 ai_panel* ai_panel) noexcept
        : show_settings_panel_(show_settings_panel)
        , show_images_panel_  (show_images_panel)
        , ai_panel_           (ai_panel)
    {}

    void main_menu_bar::draw(ui_context& ctx)
    {
        if (!ImGui::BeginMainMenuBar()) return;

        ctx.editor->draw_scene_menus();

        if (ImGui::BeginMenu("Render"))
        {
            if (ImGui::MenuItem("Save HDR (output.hdr)"))
                ctx.rt->save_hdr("output.hdr");
            if (ImGui::MenuItem("Save PNG (output.png)"))
                ctx.rt->save_png("output.png");
            ImGui::Separator();

            if (ImGui::BeginMenu("Render Mode"))
            {
                static const char* mode_labels[] = {
                    "Path trace", "Direct only",
                    "Albedo", "Normals",
                    "Instant Radiosity", "Photon Mapping"
                };
                const int current =
                    config().render_mode.load(std::memory_order_relaxed);
                for (int i = 0; i < IM_ARRAYSIZE(mode_labels); ++i)
                {
                    if (ImGui::MenuItem(mode_labels[i], nullptr, current == i))
                    {
                        if (current != i) config().set_render_mode(i);
                    }
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Sampling Strategy"))
            {
                static const char* tech_labels[] = {
                    "BSDF sampling",
                    "NEE (light sampling)",
                    "MIS (BSDF + NEE combined)"
                };
                const int current =
                    config().sampling_tech.load(std::memory_order_relaxed);
                for (int i = 0; i < IM_ARRAYSIZE(tech_labels); ++i)
                {
                    if (ImGui::MenuItem(tech_labels[i], nullptr, current == i))
                    {
                        if (current != i) config().set_sampling_tech(i);
                    }
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Adaptive Sampling"))
            {
                ImGui::TextWrapped(
                    "Routes extra rays into noisy regions");
                ImGui::Separator();

                bool use_adaptive =
                    config().use_adaptive_sampling.load(std::memory_order_relaxed);
                if (ImGui::MenuItem("Enable adaptive sampling", nullptr,
                                    &use_adaptive))
                {
                    config().set_use_adaptive_sampling(use_adaptive);
                }

                int block_size =
                    config().adaptive_block_size.load(std::memory_order_relaxed);
                if (ImGui::SliderInt("Block size (px)", &block_size, 4, 64))
                {
                    config().set_adaptive_block_size(std::max(1, block_size));
                }

                int warmup =
                    config().adaptive_warmup_spp.load(std::memory_order_relaxed);
                if (ImGui::SliderInt("Warmup SPP", &warmup, 0, 32))
                {
                    config().set_adaptive_warmup_spp(std::max(0, warmup));
                }

                int max_pp =
                    config().adaptive_max_per_pixel.load(std::memory_order_relaxed);
                if (ImGui::SliderInt("Max rays / pixel / pass", &max_pp, 1, 16))
                {
                    config().set_adaptive_max_per_pixel(std::max(1, max_pp));
                }

                ImGui::Separator();
                ImGui::TextDisabled(
                    "Worst case cost: apprx max-rays x baseline");
                ImGui::EndMenu();
            }
            ImGui::Separator();

            bool paused = config().pause_render.load(std::memory_order_relaxed);
            if (ImGui::MenuItem(paused ? "Resume Render" : "Pause Render"))
            {
                config().pause_render.store(!paused, std::memory_order_relaxed);
                if (paused) ctx.rt->clear_denoised_display();
            }
            if (ImGui::MenuItem("Reset Now"))
            {
                ctx.rt->clear_denoised_display();
                config().request_reset();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            if (ImGui::MenuItem("Settings Panel", nullptr, *show_settings_panel_))
                *show_settings_panel_ = !*show_settings_panel_;

            bool show_ed = ctx.editor->show_editor;
            if (ImGui::MenuItem("Scene Editor", nullptr, show_ed))
                ctx.editor->show_editor = !show_ed;

            bool show_hud = config().show_ui.load(std::memory_order_relaxed);
            if (ImGui::MenuItem("Show UI HUD", nullptr, show_hud))
                config().show_ui.store(!show_hud, std::memory_order_relaxed);

            if (ImGui::MenuItem("Images Panel", nullptr, *show_images_panel_))
                *show_images_panel_ = !*show_images_panel_;
            ImGui::EndMenu();
        }

        //~ denoiser
        const bool denoise_ready = ctx.rt->denoiser_available();
        const auto d_status      = ctx.rt->current_denoise_status();
        const bool d_running     =
            (d_status == render::ray_tracer::denoise_status::running);
        if (ImGui::BeginMenu("Denoiser"))
        {
            if (!denoise_ready)
                ImGui::TextDisabled("OIDN not available in this build.");

            switch (d_status)
            {
            case render::ray_tracer::denoise_status::idle:
                ImGui::TextDisabled("Status: idle");
                break;
            case render::ray_tracer::denoise_status::running:
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                                   "Status: Denoising...");
                break;
            case render::ray_tracer::denoise_status::ok:
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f),
                                   "Status: OK (%.0f ms)",
                                   ctx.rt->last_denoise_ms());
                break;
            case render::ray_tracer::denoise_status::failed:
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                                   "Status: FAILED");
                break;
            }
            ImGui::Separator();

            if (ImGui::MenuItem("Apply Denoise", "Ctrl+D", false,
                                denoise_ready && !d_running))
            {
                if (ctx.trigger_denoise) ctx.trigger_denoise();
            }
            bool showing = ctx.rt->display_denoised();
            if (ImGui::MenuItem("Showing denoised result", nullptr, showing,
                                showing))
            {
                ctx.rt->clear_denoised_display();
                config().pause_render.store(false, std::memory_order_relaxed);
            }
            ImGui::Separator();

            render::denoiser& d = ctx.rt->get_denoiser();
            ImGui::MenuItem("Use albedo AOV", nullptr, &d.use_albedo,
                            denoise_ready && !d_running);
            ImGui::MenuItem("Use normal AOV", nullptr, &d.use_normal,
                            denoise_ready && !d_running);
            ImGui::MenuItem("HDR input",      nullptr, &d.hdr_mode,
                            denoise_ready && !d_running);
            ImGui::EndMenu();
        }

        //~ effects and ai models
        if (ImGui::BeginMenu("Effects"))
        {
            ai_panel_->render_menu_section(ctx, false);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("AI Models"))
        {
            ai_panel_->render_menu_section(ctx, true);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Thread Power Control"))
        {
            const int hw = std::max(1,
                static_cast<int>(std::thread::hardware_concurrency()));
            int active = std::clamp(
                config().num_threads.load(std::memory_order_relaxed),
                1, hw);

            ImGui::Text("Logical cores detected: %d", hw);
            ImGui::TextWrapped(
                "Tom this can save your battery if you are outside you can set thread limit"
                "my code consumes power alot, as the last memory bound benchmark before instant radiosity was only"
                " 3 percentage");
            ImGui::Separator();

            if (ImGui::SliderInt("Active threads", &active, 1, hw))
            {
                config().num_threads.store(active,
                    std::memory_order_relaxed);
            }

            ImGui::SameLine();
            if (ImGui::SmallButton("Max"))
            {
                config().num_threads.store(hw, std::memory_order_relaxed);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Half"))
            {
                config().num_threads.store(std::max(1, hw / 2),
                    std::memory_order_relaxed);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Min"))
            {
                config().num_threads.store(1, std::memory_order_relaxed);
            }

            ImGui::EndMenu();
        }

        ImGui::Separator();
        ctx.editor->draw_scene_status();

        if (d_running)
        {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                               "[Denoising...]");
        }
        else if (d_status == render::ray_tracer::denoise_status::ok)
        {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f),
                               "[Denoised %.0fms]", ctx.rt->last_denoise_ms());
        }
        else if (d_status == render::ray_tracer::denoise_status::failed)
        {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                               "[Denoise FAILED]");
        }

        {
            const int   spp     = render::ctx.rt->get_spp();
            const int   tgt     = std::max(1, config().target_spp.load(
                                       std::memory_order_relaxed));
            const float frac    = std::clamp(static_cast<float>(spp)
                                  / static_cast<float>(tgt), 0.0f, 1.0f);
            const float save_w  = 60.0f;
            const float btn_w   = 70.0f;
            const float bar_w   = 180.0f;
            const float text_w  = 170.0f;
            const float pad     = 24.0f;

            ImGui::SameLine(ImGui::GetWindowWidth()
                            - (save_w + btn_w + bar_w + text_w + pad));

            ImGui::PushStyleColor(ImGuiCol_Button,
                ImVec4(0.20f, 0.35f, 0.55f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                ImVec4(0.25f, 0.45f, 0.70f, 1.0f));
            if (ImGui::Button("Save", ImVec2(save_w, 0.0f)))
            {
                ImGui::OpenPopup("##save_render_popup");
            }
            ImGui::PopStyleColor(2);
            if (ImGui::BeginPopup("##save_render_popup"))
            {
                ImGui::TextDisabled("Save current render");
                ImGui::Separator();
                if (ImGui::MenuItem("Save as PNG"))
                {
                    ctx.rt->save_png("output.png");
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::MenuItem("Save as HDR"))
                {
                    ctx.rt->save_hdr("output.hdr");
                    ImGui::CloseCurrentPopup();
                }
                if (ImGui::MenuItem("Save both PNG and HDR"))
                {
                    static int save_count = 0;
                    ++save_count;
                    std::string cp = "output_" + std::to_string(save_count) + ".";
                    ctx.rt->save_png(cp + "png");
                    ctx.rt->save_hdr(cp + ".hdr");
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            ImGui::SameLine();

            const bool paused_now =
                config().pause_render.load(std::memory_order_relaxed);
            ImGui::PushStyleColor(ImGuiCol_Button,
                paused_now ? ImVec4(0.55f, 0.20f, 0.20f, 1.0f)
                           : ImVec4(0.20f, 0.50f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                paused_now ? ImVec4(0.70f, 0.25f, 0.25f, 1.0f)
                           : ImVec4(0.25f, 0.65f, 0.30f, 1.0f));
            if (ImGui::Button(paused_now ? "Resume" : "Pause",
                              ImVec2(btn_w, 0.0f)))
            {
                config().pause_render.store(!paused_now,
                                            std::memory_order_relaxed);
                if (paused_now) ctx.rt->clear_denoised_display();
            }
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
            char eta_buf[32];
            if (ctx.eta_valid)
            {
                const float eta_sec = std::max(0.0f, ctx.eta_sec_ema);
                if (eta_sec <= 0.0f)
                {
                    std::snprintf(eta_buf, sizeof(eta_buf), "done");
                }
                else
                {
                    const int h = static_cast<int>(eta_sec / 3600.0f);
                    const int m = static_cast<int>((eta_sec - h * 3600.0f)
                                  / 60.0f);
                    const int s = static_cast<int>(eta_sec - h * 3600.0f
                                  - m * 60.0f);
                    std::snprintf(eta_buf, sizeof(eta_buf),
                                  "%02d:%02d:%02d", h, m, s);
                }
            }
            else
            {
                std::snprintf(eta_buf, sizeof(eta_buf), "--:--:--");
            }
            ImGui::Text("SPP %d/%d  ETA %s", spp, tgt, eta_buf);
            ImGui::SameLine();
            ImGui::ProgressBar(frac, ImVec2(bar_w, 0.0f));
        }

        ImGui::EndMainMenuBar();
        ctx.editor->draw_scene_popups();
    }
} // namespace fox_tracer::ui
