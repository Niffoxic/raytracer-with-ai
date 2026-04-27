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
#include "ui/image_gallery.h"

#include "config.h"

#include "utils/helper.h"
#include "utils/logger.h"
#include "utils/paths.h"
#include "render/renderer.h"

#include "framework/base.h"
#include "stb_image_write.h"

#include <filesystem>
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <imgui.h>

namespace fox_tracer::ui
{
    image_gallery::entry* image_gallery::find(std::uint64_t id)
    {
        auto it = std::ranges::find_if(
            entries_,
           [id](const entry& e)
           {
               return e.id == id;
           });

        return (it == entries_.end()) ? nullptr : &*it;
    }

    const image_gallery::entry* image_gallery::find(std::uint64_t id) const
    {
        auto it = std::ranges::find_if(
            entries_,
        [id](const entry& e)
            {
                return e.id == id;
            });
        return (it == entries_.end()) ? nullptr : &*it;
    }

    std::uint64_t image_gallery::snapshot_canvas(
        GamesEngineeringBase::Window* canvas, std::string label)
    {
        if (canvas == nullptr) return 0;
        const int w = static_cast<int>(canvas->getWidth());
        const int h = static_cast<int>(canvas->getHeight());
        if (w <= 0 || h <= 0) return 0;

        entry e;
        e.id        = next_id_++;
        e.effect    = "";
        e.state     = entry_state::snapshot;
        e.width     = w;
        e.height    = h;
        e.progress  = 1.0f;
        if (label.empty())
            e.label = helper::fmt_label("snap", ++snapshot_counter_);
        else
            e.label = std::move(label);

        e.rgb.resize(static_cast<std::size_t>(w) * h * 3);
        std::memcpy(e.rgb.data(), canvas->getBackBuffer(), e.rgb.size());
        entries_.push_back(std::move(e));
        return entries_.back().id;
    }

    std::uint64_t image_gallery::begin_effect(const std::string& effect_name,
                                              const std::uint8_t* input_rgb,
                                              const int w, const int h)
    {
        if (w <= 0 || h <= 0 || input_rgb == nullptr) return 0;

        entry e;
        e.id       = next_id_++;
        e.effect   = effect_name;
        e.state    = entry_state::in_progress;
        e.width    = w;
        e.height   = h;
        e.progress = 0.0f;
        e.label    = helper::fmt_label(effect_name.empty() ? "effect"
                                                         : effect_name,
                                     ++effect_run_counter_);
        e.rgb.assign(input_rgb,
                     input_rgb + static_cast<std::size_t>(w) * h * 3);
        entries_.push_back(std::move(e));
        return entries_.back().id;
    }

    void image_gallery::rename(std::uint64_t id, std::string new_label)
    {
        if (new_label.empty()) return;
        if (entry* e = find(id)) e->label = std::move(new_label);
    }

    void image_gallery::update_effect_partial(std::uint64_t id,
                                              const std::uint8_t* rgb,
                                              std::size_t n, float progress)
    {
        entry* e = find(id);
        if (e == nullptr) return;
        const std::size_t expect =
            static_cast<std::size_t>(e->width) * e->height * 3;
        if (n != expect)
        {
            LOG_WARN("py_effect") << "partial size mismatch!! "
                                  << id << " (" << n << " vs " << expect << ")";
            return;
        }
        std::memcpy(e->rgb.data(), rgb, n);
        e->progress = std::clamp(progress, 0.0f, 1.0f);
    }

    void image_gallery::finalize_effect(std::uint64_t id,
                                        const std::uint8_t* rgb, std::size_t n)
    {
        entry* e = find(id);
        if (e == nullptr) return;
        const std::size_t expect =
            static_cast<std::size_t>(e->width) * e->height * 3;
        if (n == expect)
        {
            std::memcpy(e->rgb.data(), rgb, n);
        }
        else
        {
            LOG_WARN("py_effect") << "done size mismatch on entry "
                                  << id << " (" << n << " vs " << expect
                                  << ") - keeping last partial";
        }
        e->state    = entry_state::done;
        e->progress = 1.0f;
    }

    void image_gallery::cancel_effect(std::uint64_t id)
    {
        if (entry* e = find(id)) e->state = entry_state::cancelled;
    }

    void image_gallery::fail_effect(std::uint64_t id, std::string msg)
    {
        if (entry* e = find(id))
        {
            e->state   = entry_state::failed;
            e->message = std::move(msg);
        }
    }

    void image_gallery::remove(std::uint64_t id)
    {
        auto it = std::ranges::find_if(
            entries_,
           [id](const entry& e)
               {
                   return e.id == id;
               });

        if (it == entries_.end()) return;
        if (viewing_id_ == id) viewing_id_ = 0;
        entries_.erase(it);
    }

    bool image_gallery::save_png(std::uint64_t id, const std::string& path) const
    {
        const entry* e = find(id);
        if (e == nullptr || e->rgb.empty()) return false;

        std::error_code ec;
        const std::filesystem::path p(path);
        if (p.has_parent_path())
        {
            std::filesystem::create_directories(p.parent_path(), ec);
        }
        return stbi_write_png(path.c_str(), e->width, e->height, 3,
                              e->rgb.data(), e->width * 3) != 0;
    }

    void image_gallery::paint_entry_to_canvas(
        const entry& e, GamesEngineeringBase::Window* canvas) const
    {
        if (canvas == nullptr || e.rgb.empty()) return;
        unsigned char* dst = canvas->getBackBuffer();
        if (dst == nullptr) return;

        const int cw = static_cast<int>(canvas->getWidth());
        const int ch = static_cast<int>(canvas->getHeight());
        if (cw <= 0 || ch <= 0) return;

        if (cw == e.width && ch == e.height)
        {
            std::memcpy(dst, e.rgb.data(),
                        static_cast<std::size_t>(cw) * ch * 3);
            return;
        }

        for (int y = 0; y < ch; ++y)
        {
            const int sy = (y * e.height) / ch;
            const std::uint8_t* src_row =
                e.rgb.data() + static_cast<std::size_t>(sy) * e.width * 3;
            unsigned char* dst_row = dst + static_cast<std::size_t>(y) * cw * 3;
            for (int x = 0; x < cw; ++x)
            {
                const int sx = (x * e.width) / cw;
                const std::uint8_t* sp = src_row + sx * 3;
                dst_row[x * 3 + 0] = sp[0];
                dst_row[x * 3 + 1] = sp[1];
                dst_row[x * 3 + 2] = sp[2];
            }
        }
    }

    void image_gallery::view(std::uint64_t id,
                             GamesEngineeringBase::Window* canvas,
                             render::ray_tracer* rt)
    {
        if (id == 0)
        {
            viewing_id_ = 0;
            if (rt != nullptr) rt->set_canvas_locked(false);

            if (paused_by_view_)
            {
                config().pause_render.store(false, std::memory_order_relaxed);
                paused_by_view_ = false;
            }
            return;
        }

        entry* e = find(id);
        if (e == nullptr) return;
        viewing_id_ = id;
        if (rt != nullptr) rt->set_canvas_locked(true);
        paint_entry_to_canvas(*e, canvas);

        if (!config().pause_render.load(std::memory_order_relaxed))
        {
            config().pause_render.store(true, std::memory_order_relaxed);
            paused_by_view_ = true;
        }
    }

    void image_gallery::refresh_canvas_if_viewing(
        GamesEngineeringBase::Window* canvas)
    {
        if (viewing_id_ == 0) return;
        const entry* e = find(viewing_id_);
        if (e == nullptr)
        {
            viewing_id_ = 0;
            return;
        }
        paint_entry_to_canvas(*e, canvas);
    }

    void image_gallery::draw_ui(bool& show,
                                GamesEngineeringBase::Window* canvas,
                                render::ray_tracer* rt,
                                const header_fn& header)
    {
        if (!show) return;

        ImGui::SetNextWindowPos (ImVec2(380.0f, 30.0f),  ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420.0f, 720.0f), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Images", &show))
        {
            ImGui::End();
            return;
        }

        if (header)
        {
            header();
            ImGui::Separator();
        }

        if (viewing_id_ == 0)
        {
            ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f),
                               "Viewing: live render (sampler running)");
        }
        else if (const entry* cur = find(viewing_id_))
        {
            ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f),
                               "Viewing: %s (sampler paused)", cur->label.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton("Back to live"))
            {
                view(0, canvas, rt);
            }
        }

        if (ImGui::Button("Snapshot live render"))
        {
            snapshot_canvas(canvas, {});
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu entr%s)",
                            entries_.size(),
                            entries_.size() == 1 ? "y" : "ies");

        ImGui::Separator();
        std::uint64_t preview_id = viewing_id_;

        if (preview_id == 0 && !entries_.empty())
        {
            preview_id = entries_.back().id;
        }
        if (const entry* pe = find(preview_id); pe != nullptr && !pe->rgb.empty())
        {
            ImGui::TextDisabled("Preview: %s (%dx%d)",
                                pe->label.c_str(), pe->width, pe->height);

            const float panel_w = ImGui::GetContentRegionAvail().x;
            const float aspect  = (pe->height > 0)
                ? static_cast<float>(pe->width) / static_cast<float>(pe->height)
                : 1.0f;

            const float w = std::min(panel_w, 400.0f);
            const float h = (aspect > 0.0f) ? (w / aspect) : w;

            const ImVec2 p0 = ImGui::GetCursorScreenPos();
            const ImVec2 p1 = ImVec2(p0.x + w, p0.y + h);

            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(p0, p1, IM_COL32(20, 20, 20, 255));

            constexpr int kGrid = 64;

            const int cells_x = std::min(kGrid,
                                         static_cast<int>(std::max(8.0f, w / 4.0f)));
            const int cells_y = std::min(kGrid,
                                         static_cast<int>(std::max(8.0f, h / 4.0f)));

            const float cw = w / static_cast<float>(cells_x);
            const float ch = h / static_cast<float>(cells_y);

            for (int j = 0; j < cells_y; ++j)
            {
                const int sy = (j * pe->height) / cells_y;

                for (int i = 0; i < cells_x; ++i)
                {
                    const int sx = (i * pe->width) / cells_x;

                    const std::uint8_t* sp =
                        pe->rgb.data() +
                        (static_cast<std::size_t>(sy) * pe->width + sx) * 3;

                    const ImU32 col = IM_COL32(sp[0], sp[1], sp[2], 255);
                    const ImVec2 c0(p0.x + i * cw,       p0.y + j * ch);
                    const ImVec2 c1(p0.x + (i + 1) * cw, p0.y + (j + 1) * ch);

                    dl->AddRectFilled(c0, c1, col);
                }
            }
            dl->AddRect(p0, p1, IM_COL32(120, 120, 120, 255));
            ImGui::Dummy(ImVec2(w, h));

            if (ImGui::Button("View##preview", ImVec2(70.0f, 0.0f)))
            {
                view(preview_id, canvas, rt);
            }
            ImGui::SameLine();
            if (ImGui::Button("Rename##preview", ImVec2(85.0f, 0.0f)))
            {
                std::snprintf(rename_buf_, sizeof(rename_buf_),
                              "%s", pe->label.c_str());
                rename_target_id_ = preview_id;
                ImGui::OpenPopup("rename_entry");
            }
            ImGui::SameLine();
            if (ImGui::Button("Save PNG##preview", ImVec2(110.0f, 0.0f)))
            {
                const std::string out =
                    (std::filesystem::path(paths::resolve("saved"))
                        / (pe->label + ".png")).string();
                if (save_png(preview_id, out))
                    LOG_INFO("gallery") << "saved " << out;
                else
                    LOG_WARN("gallery") << "failed to save " << out;
            }
            ImGui::Separator();
        }

        if (entries_.empty())
        {
            ImGui::TextDisabled("No images yet capture a snapshot or run an effect please");
            ImGui::End();
            return;
        }

        std::vector<std::uint64_t> ids;
        ids.reserve(entries_.size());
        for (const entry& e : entries_) ids.push_back(e.id);

        std::uint64_t to_remove = 0;

        for (std::uint64_t id : ids)
        {
            entry* e = find(id);
            if (e == nullptr) continue;

            ImGui::PushID(static_cast<int>(id));

            const char* state_str = "snapshot";
            ImVec4 state_col(0.7f, 0.7f, 0.7f, 1.0f);
            switch (e->state)
            {
            case entry_state::snapshot:
                state_str = "snapshot";
                state_col = ImVec4(0.6f, 0.7f, 1.0f, 1.0f);
                break;
            case entry_state::in_progress:
                state_str = "running";
                state_col = ImVec4(1.0f, 0.85f, 0.2f, 1.0f);
                break;
            case entry_state::done:
                state_str = "done";
                state_col = ImVec4(0.3f, 1.0f, 0.4f, 1.0f);
                break;
            case entry_state::cancelled:
                state_str = "cancelled";
                state_col = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                break;
            case entry_state::failed:
                state_str = "failed";
                state_col = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                break;
            }

            const bool is_viewing = (viewing_id_ == id);
            if (is_viewing)
            {
                ImGui::PushStyleColor(ImGuiCol_Header,
                                      ImVec4(0.2f, 0.4f, 0.2f, 1.0f));
            }
            ImGui::Selectable(e->label.c_str(), is_viewing,
                              ImGuiSelectableFlags_AllowDoubleClick);
            if (is_viewing) ImGui::PopStyleColor();

            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
            {
                view(id, canvas, rt);
            }

            ImGui::SameLine();
            ImGui::TextColored(state_col, "[%s]", state_str);

            if (!e->effect.empty())
            {
                ImGui::SameLine();
                ImGui::TextDisabled(" %s @ %dx%d",
                                    e->effect.c_str(), e->width, e->height);
            }
            else
            {
                ImGui::SameLine();
                ImGui::TextDisabled(" %dx%d", e->width, e->height);
            }

            if (e->state == entry_state::in_progress)
            {
                ImGui::ProgressBar(e->progress, ImVec2(-1.0f, 4.0f));
            }
            if (e->state == entry_state::failed && !e->message.empty())
            {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                                   "!  %s", e->message.c_str());
            }

            const bool can_view = !e->rgb.empty();
            if (ImGui::Button("View", ImVec2(60.0f, 0.0f)) && can_view)
            {
                view(id, canvas, rt);
            }
            ImGui::SameLine();
            if (ImGui::Button("Rename", ImVec2(70.0f, 0.0f)))
            {
                std::snprintf(rename_buf_, sizeof(rename_buf_),
                              "%s", e->label.c_str());
                rename_target_id_ = id;
                ImGui::OpenPopup("rename_entry");
            }
            ImGui::SameLine();
            if (ImGui::Button("Save PNG", ImVec2(80.0f, 0.0f)) && can_view)
            {
                const std::string out =
                    (std::filesystem::path(paths::resolve("saved"))
                        / (e->label + ".png")).string();
                if (save_png(id, out))
                    LOG_INFO("gallery") << "saved " << out;
                else
                    LOG_WARN("gallery") << "failed to save " << out;
            }
            ImGui::SameLine();
            if (ImGui::Button("Drop", ImVec2(60.0f, 0.0f)))
            {
                to_remove = id;
            }

            ImGui::Separator();
            ImGui::PopID();
        }

        if (to_remove != 0) remove(to_remove);

        if (ImGui::BeginPopupModal("rename_entry", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::TextDisabled("Renaming entry id %llu",
                                static_cast<unsigned long long>(rename_target_id_));
            ImGui::SetNextItemWidth(280.0f);
            const bool committed = ImGui::InputText(
                "##rename_input", rename_buf_, sizeof(rename_buf_),
                ImGuiInputTextFlags_EnterReturnsTrue);
            const bool clicked_ok = ImGui::Button("Rename", ImVec2(110.0f, 0.0f));
            ImGui::SameLine();
            const bool clicked_cancel = ImGui::Button("Cancel", ImVec2(110.0f, 0.0f));

            if (committed || clicked_ok)
            {
                rename(rename_target_id_, rename_buf_);
                rename_target_id_ = 0;
                ImGui::CloseCurrentPopup();
            }
            else if (clicked_cancel)
            {
                rename_target_id_ = 0;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::End();
    }
} // namespace fox_tracer
