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
#ifndef RAYTRACER_WITH_AI_INDEPENDENT_H
#define RAYTRACER_WITH_AI_INDEPENDENT_H

#include "pixel_sampler.h"

namespace fox_tracer::sampling
{
    //~ same as mt tho added for completeness: uniform random
    // unbiased, fast(since parallel is easy because of no shared state)
    // (fox tracer debug default)
    class independent_sampler final: public pixel_sampler
    {
    public:
        explicit independent_sampler(unsigned int seed = 1);

        float next() override;

        void start_pixel    (int px, int py, int sample_index) override;
        void reset_with_seed(unsigned int seed)                override;

    private:
        std::mt19937                          generator_;
        std::uniform_real_distribution<float> dist_{0.0f, 1.0f};
    };
} // fox_tracer::sampling

#endif //RAYTRACER_WITH_AI_INDEPENDENT_H
