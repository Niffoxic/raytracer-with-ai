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
#ifndef RAYTRACER_WITH_AI_HALTON_SAMPLER_H
#define RAYTRACER_WITH_AI_HALTON_SAMPLER_H

#include "pixel_sampler.h"
#include <cstdint>

namespace fox_tracer::sampling
{
    //~ progressive, low-discrepancy
    class halton_sampler final : public pixel_sampler
    {
    public:
        explicit halton_sampler(bool          scrambling     = true,
                                int           max_dimensions = 256,
                                unsigned int  seed           = 1);

        float next() override;

        void start_pixel    (int px, int py,
                             int sample_index)  override;
        void start_dimension(int dim)           override;
        void reset_with_seed(unsigned int seed) override;

    private:
        float scrambled_radical_inverse(int dim, std::uint64_t i) const;

        bool          scrambling_;
        int           max_dimensions_;
        std::uint64_t sample_index_{0};
        int           dim_{0};
        unsigned int  seed_{1};
    };
} // fox_tracer::sampling

#endif //RAYTRACER_WITH_AI_HALTON_SAMPLER_H
