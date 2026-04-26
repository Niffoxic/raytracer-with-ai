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
#ifndef RAYTRACER_WITH_AI_DENOISER_H
#define RAYTRACER_WITH_AI_DENOISER_H

#include "framework/core.h"

namespace fox_tracer::render
{
    class denoiser
    {
    public:
         denoiser() noexcept;
        ~denoiser();

        denoiser(const denoiser&)            = delete;
        denoiser& operator=(const denoiser&) = delete;
        denoiser(denoiser&&)                 = delete;
        denoiser& operator=(denoiser&&)      = delete;

        bool ensure_ready();

        [[nodiscard]] bool available() const noexcept { return device_ != nullptr; }

        bool denoise(const color* color_in,
                     const color* albedo_in,
                     const color* normal_in,
                     color*       color_out,
                     int          width,
                     int          height,
                     bool         hdr);

        bool use_albedo{true};
        bool use_normal{true};
        bool hdr_mode  {true};

    private:
        void* device_{nullptr};
    };
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_DENOISER_H
