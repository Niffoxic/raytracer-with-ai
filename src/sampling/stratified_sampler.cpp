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
#include "sampler/stratified_sampler.h"

namespace
{
    std::uint32_t hash_combine(std::uint32_t a, std::uint32_t b) noexcept
    {
        a ^= b + 0x9e3779b9u + (a << 6) + (a >> 2);
        return a;
    }

    //~ straight outta PermutationElement xD
    int permute(const int i, const int n, std::uint32_t p) noexcept
    {
        std::uint32_t w = static_cast<std::uint32_t>(n - 1);
        w |= w >> 1; w |= w >> 2; w |= w >> 4;
        w |= w >> 8; w |= w >> 16;

        std::uint32_t v = static_cast<std::uint32_t>(i);
        do {
            v ^= p;
            v *= 0xe170893du;
            v ^= p >> 16;
            v ^= (v & w) >> 4;
            v ^= p >> 8;
            v *= 0x0929eb3fu;
            v ^= p >> 23;
            v ^= (v & w) >> 1;
            v *= 1u | p >> 27;
            v *= 0x6935fa69u;
            v ^= (v & w) >> 11;
            v *= 0x74dcb303u;
            v ^= (v & w) >> 2;
            v *= 0x9e501cc3u;
            v ^= (v & w) >> 2;
            v *= 0xc860a3dfu;
            v &= w;
            v ^= v >> 5;
        } while (v >= static_cast<std::uint32_t>(n));
        return static_cast<int>((v + p) % static_cast<std::uint32_t>(n));
    }
}

fox_tracer::sampling::stratified_sampler::stratified_sampler(
    const int samples_per_axis, const unsigned int seed)
    : samples_per_axis_(std::max(1, samples_per_axis))
{
    generator_.seed(seed);
}

float fox_tracer::sampling::stratified_sampler::next()
{
    const int N = samples_per_axis_ * samples_per_axis_;
    if (dim_ < 2 && sample_index_ < N)
    {
        const int permuted_cell = permute(sample_index_, N,
                                          pixel_hash_ ^ 0x68bc21ebu);

        const int sx = permuted_cell % samples_per_axis_;
        const int sy = permuted_cell / samples_per_axis_;

        const float jitter = dist_(generator_);
        const float coord  = (dim_ == 0)
            ? (sx + jitter) / static_cast<float>(samples_per_axis_)
            : (sy + jitter) / static_cast<float>(samples_per_axis_);
        ++dim_;
        return coord;
    }

    ++dim_;
    return dist_(generator_);
}

void fox_tracer::sampling::stratified_sampler::start_pixel(
    const int px, const int py, const int sample_index)
{
    const auto hpy  = static_cast<std::uint32_t>(py) * 0x85ebca6bu;
    const auto inner = hash_combine(static_cast<std::uint32_t>(px), hpy);
    const auto hsi  = static_cast<std::uint32_t>(sample_index) * 0xc2b2ae35u;
    pixel_hash_   = hash_combine(inner,hsi);
    sample_index_ = sample_index;
    dim_          = 0;
}

void fox_tracer::sampling::stratified_sampler::reset_with_seed(unsigned int seed)
{
    generator_.seed(seed);
    sample_index_ = 0;
    dim_          = 0;
}
