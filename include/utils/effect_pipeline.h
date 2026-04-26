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
#ifndef RAYTRACER_WITH_AI_EFFECT_PIPELINE_H
#define RAYTRACER_WITH_AI_EFFECT_PIPELINE_H

#include "ui/image_gallery.h"
#include "effect_runner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace GamesEngineeringBase { class Window; }

namespace fox_tracer::utility
{
    class effect_pipeline
    {
    public:
         effect_pipeline() noexcept = default;
        ~effect_pipeline() = default;

        effect_pipeline(const effect_pipeline&)            = delete;
        effect_pipeline& operator=(const effect_pipeline&) = delete;

        void start   ();
        void shutdown();
        void poll    ();

        void launch(const std::string& effect_name, const GamesEngineeringBase::Window* window,
                    bool& show_images_panel);

        void rescan ();
        void restart();

        effect_runner&     runner ()  noexcept { return runner_; }
        ui::image_gallery& gallery()  noexcept { return gallery_; }

        const std::string* phase_label_ptr   () const noexcept { return &phase_label_; }
        const float*       phase_progress_ptr() const noexcept { return &phase_progress_; }
        const std::string* device_ptr        () const noexcept { return &device_; }

    private:
        void reset_pending_state();

        effect_runner     runner_;
        ui::image_gallery gallery_;

        bool started_{false};

        std::uint64_t             pending_entry_id_{0};
        std::string               pending_effect_name_;
        std::string               pending_label_;
        int                       pending_w_{0};
        int                       pending_h_{0};
        std::vector<std::uint8_t> pending_input_rgb_;
        bool                      pending_infer_after_load_{false};

        std::string phase_label_;
        float       phase_progress_{0.0f};
        std::string device_;
    };
} // namespace fox_tracer::utility

#endif //RAYTRACER_WITH_AI_EFFECT_PIPELINE_H
