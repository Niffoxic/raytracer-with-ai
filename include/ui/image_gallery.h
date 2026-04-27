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
#ifndef RAYTRACER_WITH_AI_IMAGE_GALLERY_H
#define RAYTRACER_WITH_AI_IMAGE_GALLERY_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace GamesEngineeringBase { class Window; }

namespace fox_tracer::render
{
    class ray_tracer;
}

namespace fox_tracer::ui
{
    class image_gallery
    {
    public:
        enum class entry_state : int
        {
            snapshot    = 0,
            in_progress = 1,
            done        = 2,
            cancelled   = 3,
            failed      = 4
        };

        struct entry
        {
            std::uint64_t               id{0};
            std::string                 label;
            std::string                 effect;
            entry_state                 state{entry_state::snapshot};
            int                         width{0};
            int                         height{0};
            float                       progress{0.0f};
            std::string                 message;
            std::vector<std::uint8_t>   rgb;
        };

        image_gallery() noexcept = default;

        std::uint64_t snapshot_canvas(GamesEngineeringBase::Window* canvas,
                                      std::string label);

        std::uint64_t begin_effect(const std::string& effect_name,
                                   const std::uint8_t* input_rgb,
                                   int w, int h);

        void update_effect_partial(std::uint64_t id,
                                   const std::uint8_t* rgb, std::size_t n,
                                   float progress);

        void finalize_effect(std::uint64_t id,
                             const std::uint8_t* rgb, std::size_t n);

        void cancel_effect(std::uint64_t id);
        void fail_effect(std::uint64_t id, std::string msg);

        void remove(std::uint64_t id);

        bool save_png(std::uint64_t id, const std::string& path) const;

        [[nodiscard]] std::uint64_t viewing_id() const noexcept
        {
            return viewing_id_;
        }

        void view(std::uint64_t id, GamesEngineeringBase::Window* canvas,
                  render::ray_tracer* rt);

        void refresh_canvas_if_viewing(GamesEngineeringBase::Window* canvas);

        using header_fn = std::function<void()>;
        void draw_ui(bool& show,
                     GamesEngineeringBase::Window* canvas,
                     render::ray_tracer* rt,
                     const header_fn& header = {});

        void rename(std::uint64_t id, std::string new_label);

        [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    private:
              entry* find(std::uint64_t id);
        const entry* find(std::uint64_t id) const;

        void paint_entry_to_canvas(const entry& e,
                                   GamesEngineeringBase::Window* canvas) const;

        std::vector<entry>  entries_;
        std::uint64_t       next_id_    {1};
        std::uint64_t       viewing_id_ {0};
        int                 snapshot_counter_   {0};
        int                 effect_run_counter_ {0};

        bool                paused_by_view_{false};

        std::uint64_t       rename_target_id_{0};
        char                rename_buf_[128]{};
    };
} // namespace fox_tracer::ui

#endif //RAYTRACER_WITH_AI_IMAGE_GALLERY_H
