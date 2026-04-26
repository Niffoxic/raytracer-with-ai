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
#include "ui/ui_denoise.h"
#include "ui/ui_context.h"

#include "render/renderer.h"

#include <imgui.h>
#include <algorithm>
#include <string>

namespace fox_tracer::ui
{
    void denoise_component::draw(ui_context& ctx)
    {
        const int st = static_cast<int>(ctx.rt->current_denoise_status());
        const bool running =
            (st == static_cast<int>(render::ray_tracer::denoise_status::running));
        const bool finished =
            (st == static_cast<int>(render::ray_tracer::denoise_status::ok))
         || (st == static_cast<int>(render::ray_tracer::denoise_status::failed));

        if (st != last_seen_status_)
        {
            last_seen_status_ = st;
            if (finished)
            {
                toast_tp_     = std::chrono::steady_clock::now();
                toast_active_ = true;
            }
        }

        constexpr float persist_sec = 6.0f;
        float toast_age_sec = 0.0f;
        if (toast_active_)
        {
            toast_age_sec = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - toast_tp_).count();
            if (!running && toast_age_sec > persist_sec)
                toast_active_ = false;
        }

        const bool draw_toast = running || toast_active_;
        if (!draw_toast) return;

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const float pad = 12.0f;
        ImGui::SetNextWindowPos(
            ImVec2(vp->WorkPos.x + vp->WorkSize.x - pad,
                   vp->WorkPos.y + pad),
            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
        ImGui::SetNextWindowBgAlpha(running ? 0.85f : std::max(0.25f,
            0.9f - (toast_age_sec / persist_sec) * 0.65f));

        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration       |
            ImGuiWindowFlags_AlwaysAutoResize   |
            ImGuiWindowFlags_NoSavedSettings    |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav              |
            ImGuiWindowFlags_NoMove;

        if (ImGui::Begin("##denoise_component", nullptr, flags))
        {
            if (running)
            {
                constexpr char frames[] = { '|', '/', '-', '\\' };
                const double now_sec = ImGui::GetTime();
                const int fi = static_cast<int>(now_sec * 10.0) & 3;
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                                   "%c Denoising...", frames[fi]);
            }
            else
            {
                const bool ok =
                    (st == static_cast<int>(render::ray_tracer::denoise_status::ok));
                ImGui::TextColored(ok ? ImVec4(0.3f, 1.0f, 0.4f, 1.0f)
                                      : ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                                   ok ? "Denoise complete" : "Denoise FAILED");
                const std::string msg = ctx.rt->last_denoise_message();
                if (!msg.empty())
                {
                    ImGui::TextDisabled("%s", msg.c_str());
                }
            }
        }
        ImGui::End();
    }
} // namespace fox_tracer::ui
