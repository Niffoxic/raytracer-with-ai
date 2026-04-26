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
#ifndef RAYTRACER_WITH_AI_SCENE_HOST_H
#define RAYTRACER_WITH_AI_SCENE_HOST_H

#include "scene_editor.h"

#include <cstdint>
#include <string>

namespace fox_tracer::scene
{
    class container;
    class ray_tracer;

    class scene_host
    {
    public:
        scene_host()  noexcept = default;
        ~scene_host();

        scene_host(const scene_host&)            = delete;
        scene_host& operator=(const scene_host&) = delete;

        bool init(const std::string& scene_name,
                  const std::string& assets_root,
                  int width, int height);

        void check_pending_reset (ray_tracer& rt);
        void check_pending_editor(ray_tracer& rt);

        void release();

        [[nodiscard]] container*    current   () const  noexcept { return current_; }
        [[nodiscard]] scene_editor& editor    ()        noexcept { return editor_; }
        [[nodiscard]] bool          scene_idle() const  noexcept
        { return scene_idle_until_loaded_; }

    private:
        container*    current_{nullptr};
        scene_editor  editor_;
        std::uint32_t cached_reset_gen_{0};
        bool          scene_idle_until_loaded_{false};
    };
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_SCENE_HOST_H
