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
#include "ui/ai_panel.h"
#include "ui/ui_context.h"
#include "utils/effect_runner.h"


#include <windows.h>
#include <imgui.h>
#include <algorithm>
#include <string>
#include <vector>

// TODO: add key gen for serverless arch apis
namespace fox_tracer::ui
{
    namespace
    {
        const char* const kTorchBackendLabels[] =
        {
            "Auto (CUDA if detected, else CPU)",
            "CPU only (whl/cpu)",
            "CUDA 11.8 (whl/cu118)",
            "CUDA 12.1 (whl/cu121) - legacy",
            "CUDA 12.4 (whl/cu124) - recommended",
            "CUDA 12.6 (whl/cu126)",
            "CUDA 12.8 (whl/cu128)",
            "ROCm 6.2 (whl/rocm6.2, Linux)",
            "Custom index URL...",
        };
        const char* const kTorchBackendUrls[] =
        {
            "",
            "https://download.pytorch.org/whl/cpu",
            "https://download.pytorch.org/whl/cu118",
            "https://download.pytorch.org/whl/cu121",
            "https://download.pytorch.org/whl/cu124",
            "https://download.pytorch.org/whl/cu126",
            "https://download.pytorch.org/whl/cu128",
            "https://download.pytorch.org/whl/rocm6.2",
            "",
        };
        constexpr int kTorchBackendAuto   = 0;
        constexpr int kTorchBackendCustom = static_cast<int>(std::size(kTorchBackendLabels)) - 1;

        bool is_ai_effect(const std::string& name)
        {
            return name.size() >= 3 && name[0] == 'a' &&
                   name[1] == 'i' && name[2] == '_';
        }
    } // namespace

    ai_panel::ai_panel(bool* show_images_panel) noexcept
        : show_images_panel_(show_images_panel)
    {}

    void ai_panel::draw(ui_context& ctx)
    {
        if (open_cuda_confirm_)
        {
            ImGui::OpenPopup("CUDA not detected");
            open_cuda_confirm_ = false;
        }

        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing,
                                ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("CUDA not detected!", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                               "Tom you need to install CUDA or the app will be hella slow");
            ImGui::Spacing();
            ImGui::TextWrapped(
                "Torch will install the CPU only wheel!. Inference on "
                "CPU is somewhere round 10-50x slower than on a GPU");
            ImGui::Spacing();
            ImGui::TextWrapped(
                "To use your NVIDIA GPU, install CUDA 12.1 (or newer) "
                "I personally recommended 12.4 it has everything my program needs");
            ImGui::Separator();
            if (ImGui::Button("Install CPU-only anyway",
                              ImVec2(200.0f, 0.0f)))
            {
                if (ctx.install_ai_deps_now)
                    ctx.install_ai_deps_now(std::string{});
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120.0f, 0.0f)))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    void ai_panel::render_menu_section(ui_context& ctx, const bool ai_only)
    {
        utility::effect_runner& fx = *ctx.fx_runner;
        const auto fx_status = fx.current_status();
        const bool fx_alive  = fx.alive();
        const bool fx_busy   =
            (fx_status == utility::effect_runner::status::loading ||
             fx_status == utility::effect_runner::status::running);

        if (!fx_alive)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "Python worker not running.");
            const std::string err = fx.last_error();
            if (!err.empty())
                ImGui::TextWrapped("%s", err.c_str());
            ImGui::TextDisabled(
                "Drop a .venv into assets/scripts/ use Create .venv");
            ImGui::TextDisabled(
                "below) set FOX_TRACER_PYTHON or install python on PATH");
            if (ImGui::MenuItem("Restart worker"))
            {
                if (ctx.restart_worker) ctx.restart_worker();
            }
            ImGui::Separator();
        }
        else
        {
            switch (fx_status)
            {
            case utility::effect_runner::status::idle:
                ImGui::TextDisabled("Status: idle");
                break;
            case utility::effect_runner::status::loading:
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                                   "Status: loading '%s'...",
                                   fx.active_effect_name().c_str());
                break;
            case utility::effect_runner::status::running:
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                                   "Status: running '%s'",
                                   fx.active_effect_name().c_str());
                break;
            case utility::effect_runner::status::ok:
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f),
                                   "Status: last run OK");
                break;
            case utility::effect_runner::status::failed:
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                                   "Status: FAILED");
                {
                    const std::string err = fx.last_error();
                    if (!err.empty())
                        ImGui::TextWrapped("%s", err.c_str());
                }
                break;
            }

            if (ctx.fx_phase_label && !ctx.fx_phase_label->empty())
            {
                ImGui::TextColored(ImVec4(0.75f, 0.85f, 1.0f, 1.0f),
                                   "Phase: %s", ctx.fx_phase_label->c_str());
                const float prog = ctx.fx_phase_progress
                                    ? *ctx.fx_phase_progress : 0.0f;
                if (prog > 0.0f)
                {
                    ImGui::ProgressBar(std::clamp(prog, 0.0f, 1.0f),
                                       ImVec2(220.0f, 0.0f));
                }
            }

            if (ai_only)
            {
                if (ctx.fx_device && !ctx.fx_device->empty())
                {
                    const std::string& dev = *ctx.fx_device;
                    const bool on_gpu =
                        (dev.rfind("cuda", 0) == 0 ||
                         dev == "mps"          ||
                         dev.rfind("xpu",  0) == 0);
                    ImGui::TextColored(
                        on_gpu ? ImVec4(0.35f, 1.0f, 0.6f, 1.0f)
                               : ImVec4(1.0f, 0.65f, 0.35f, 1.0f),
                        "Torch device: %s%s",
                        dev.c_str(),
                        on_gpu ? " (GPU)" : " (CPU)");
                }
                const std::string py = fx.picked_python();
                if (!py.empty())
                    ImGui::TextDisabled("Python: %s", py.c_str());
            }
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Rescan", nullptr, false, fx_alive))
        {
            if (ctx.rescan_effects) ctx.rescan_effects();
        }

        const auto effects_all = fx.cached_effects();
        std::vector<std::string> effects;
        effects.reserve(effects_all.size());
        for (const std::string& n : effects_all)
        {
            if (is_ai_effect(n) == ai_only) effects.push_back(n);
        }
        if (effects.empty())
        {
            ImGui::TextDisabled(ai_only ? "No ai_* scripts discovered."
                                        : "No effects discovered.");
        }
        else
        {
            ImGui::Separator();
            for (const std::string& name : effects)
            {
                if (ImGui::MenuItem(name.c_str(), nullptr, false,
                                    fx_alive && !fx_busy))
                {
                    if (ctx.launch_effect) ctx.launch_effect(name);
                }
            }
        }

        ImGui::Separator();
        if (ImGui::MenuItem("Cancel running", nullptr, false,
                            fx_alive && fx_busy))
        {
            fx.request_cancel();
        }
        const bool fx_has_loaded =
            fx_alive && !fx_busy && !fx.active_effect_name().empty();
        if (ImGui::MenuItem("Unload effect", nullptr, false, fx_has_loaded))
        {
            fx.request_unload();
        }
        if (ImGui::MenuItem("Restart worker"))
        {
            if (ctx.restart_worker) ctx.restart_worker();
        }
        if (show_images_panel_ &&
            ImGui::MenuItem("Open Images Panel", nullptr, *show_images_panel_))
        {
            *show_images_panel_ = !*show_images_panel_;
        }

        if (ai_only)
        {
            ImGui::Separator();
            const bool        bs_busy   = ctx.bootstrap_busy
                                            ? ctx.bootstrap_busy->load()
                                            : false;
            const std::string venv_py   = ctx.resolve_venv_python
                                            ? ctx.resolve_venv_python()
                                            : std::string{};
            const std::string worker_py = fx.picked_python();

            if (venv_py.empty())
            {
                ImGui::TextDisabled(
                    "No local .venv yet create one to keep AI deps");
                ImGui::TextDisabled(
                    "out of your system Python.");
            }
            else
            {
                ImGui::TextColored(ImVec4(0.55f, 1.0f, 0.55f, 1.0f),
                                   "Venv: %s", venv_py.c_str());
            }

            if (!venv_py.empty() && !worker_py.empty() &&
                venv_py != worker_py)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                    "Worker is using a different Python");
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                    "than the venv - click Restart worker after");
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                    "installing deps.");
            }

            if (ImGui::MenuItem("Create local .venv",
                                "python -m venv --copies "
                                "assets/scripts/.venv",
                                false, !bs_busy))
            {
                if (ctx.create_venv) ctx.create_venv();
            }

            ImGui::Separator();
            ImGui::TextDisabled("Torch wheel source");
            ImGui::SetNextItemWidth(260.0f);
            ImGui::Combo("##torch_backend",
                         &torch_backend_idx_,
                         kTorchBackendLabels,
                         IM_ARRAYSIZE(kTorchBackendLabels));
            if (torch_backend_idx_ == kTorchBackendCustom)
            {
                ImGui::SetNextItemWidth(260.0f);
                ImGui::InputTextWithHint(
                    "##torch_custom_url",
                    "look out torch website",
                    torch_custom_url_,
                    sizeof(torch_custom_url_));
                ImGui::TextDisabled(
                    "e.g. whl/cu126, whl/nightly/cu128, whl/rocm6.2");
            }
            else if (torch_backend_idx_ != kTorchBackendAuto)
            {
                ImGui::TextDisabled(
                    "URL: %s", kTorchBackendUrls[torch_backend_idx_]);
            }

            const char* install_hint =
                venv_py.empty()
                    ? "pip install ... (into worker interpreter)"
                    : "pip install ... (into local .venv)";
            if (ImGui::MenuItem("Install AI deps", install_hint,
                                false, !bs_busy))
            {
                on_install_clicked(ctx);
            }

            std::string summary;
            if (ctx.bootstrap_mtx && ctx.bootstrap_last_summary)
            {
                std::lock_guard<std::mutex> lk(*ctx.bootstrap_mtx);
                summary = *ctx.bootstrap_last_summary;
            }
            if (bs_busy)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                                   "Bootstrap: running...");
            }
            else if (!summary.empty())
            {
                ImGui::TextDisabled("Last: %s", summary.c_str());
            }
        }
    }

    void ai_panel::on_install_clicked(ui_context& ctx)
    {
        if (!ctx.install_ai_deps_now) return;

        char env_buf[1024];
        DWORD env_n = GetEnvironmentVariableA(
            "FOX_TRACER_TORCH_INDEX", env_buf, 1024);
        if (env_n > 0 && env_n < 1024)
        {
            ctx.install_ai_deps_now(std::string(env_buf));
            return;
        }

        if (torch_backend_idx_ == kTorchBackendCustom)
        {
            ctx.install_ai_deps_now(std::string(torch_custom_url_));
            return;
        }

        if (torch_backend_idx_ == kTorchBackendAuto)
        {
            if (ctx.refresh_cuda_detection) ctx.refresh_cuda_detection();
            const bool cuda_ok = ctx.cuda_available && ctx.cuda_available();
            if (cuda_ok)
            {
                ctx.install_ai_deps_now(
                    "https://download.pytorch.org/whl/cu124");
            }
            else
            {
                open_cuda_confirm_ = true;
            }
            return;
        }

        ctx.install_ai_deps_now(
            std::string(kTorchBackendUrls[torch_backend_idx_]));
    }

    void ai_panel::render_status_header(const ui_context& ctx) const
    {
        utility::effect_runner& fx = *ctx.fx_runner;
        const auto  fx_status   = fx.current_status();
        const bool  fx_alive    = fx.alive();
        const auto  active_name = fx.active_effect_name();
        const bool  is_ai_active = is_ai_effect(active_name);
        if (!fx_alive || !is_ai_active) return;

        ImGui::TextColored(ImVec4(0.75f, 0.85f, 1.0f, 1.0f),
                           "AI: %s", active_name.c_str());
        switch (fx_status)
        {
        case utility::effect_runner::status::idle:
            ImGui::TextDisabled("Status: idle"); break;
        case utility::effect_runner::status::loading:
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                               "Status: loading..."); break;
        case utility::effect_runner::status::running:
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                               "Status: running"); break;
        case utility::effect_runner::status::ok:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f),
                               "Status: last run OK"); break;
        case utility::effect_runner::status::failed:
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                               "Status: FAILED"); break;
        }

        if (ctx.fx_phase_label && !ctx.fx_phase_label->empty())
        {
            ImGui::TextColored(ImVec4(0.75f, 0.85f, 1.0f, 1.0f),
                               "Phase: %s", ctx.fx_phase_label->c_str());
            const float prog = ctx.fx_phase_progress
                                ? *ctx.fx_phase_progress : 0.0f;
            if (prog > 0.0f)
            {
                ImGui::ProgressBar(std::clamp(prog, 0.0f, 1.0f),
                                   ImVec2(-1.0f, 0.0f));
            }
        }

        if (ctx.fx_device && !ctx.fx_device->empty())
        {
            const std::string& dev = *ctx.fx_device;
            const bool on_gpu =
                (dev.rfind("cuda", 0) == 0 ||
                 dev == "mps"          ||
                 dev.rfind("xpu",  0) == 0);
            ImGui::TextColored(
                on_gpu ? ImVec4(0.35f, 1.0f, 0.6f, 1.0f)
                       : ImVec4(1.0f, 0.65f, 0.35f, 1.0f),
                "Torch: %s%s", dev.c_str(),
                on_gpu ? " (GPU)" : " (CPU)");
        }

        if (ImGui::CollapsingHeader("Worker log",
                                    ImGuiTreeNodeFlags_DefaultOpen))
        {
            const auto lines = fx.recent_log_lines(80);
            ImGui::BeginChild("py_worker_log",
                              ImVec2(0.0f, 160.0f),
                              true,
                              ImGuiWindowFlags_HorizontalScrollbar);
            for (const std::string& ln : lines)
            {
                ImGui::TextUnformatted(ln.c_str());
            }
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f)
                ImGui::SetScrollHereY(1.0f);
            ImGui::EndChild();
        }
    }
} // namespace fox_tracer::ui
