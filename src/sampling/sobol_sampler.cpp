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
#include "sampler/sobol_sampler.h"
#include <array>

namespace
{
    //~ TODO: Load the full 21201 dimension if I have time later (could read from files)
    //~ (camera 2 + lens 2 + bounces × (bsdf 2 + light 3))
    //~ TODO: switch to embed
    constexpr int FOX_SOBOL_VERBATIM_DIMS = 32;
    constexpr int FOX_SOBOL_BITS          = 32;

    //~ One row of the Joe Kuo file
    struct joe_kuo_row
    {
        int                          s;
        std::uint32_t                a;
        std::array<std::uint32_t, 8> m;
    };

    constexpr joe_kuo_row foxJoeKuo[FOX_SOBOL_VERBATIM_DIMS] = {
        { 1, 0, {1, 0, 0, 0, 0, 0, 0, 0} },          // dim  1
        { 2, 1, {1, 3, 0, 0, 0, 0, 0, 0} },          // dim  2
        { 3, 1, {1, 3, 1, 0, 0, 0, 0, 0} },          // dim  3
        { 3, 2, {1, 1, 1, 0, 0, 0, 0, 0} },          // 4
        { 4, 1, {1, 1, 3, 3, 0, 0, 0, 0} },          // 5
        { 4, 4, {1, 3, 5, 13, 0, 0, 0, 0} },         // 6
        { 5, 2, {1, 1, 5, 5, 17, 0, 0, 0} },         // 7
        { 5, 4, {1, 1, 5, 5, 5, 0, 0, 0} },          // .. so on
        { 5, 7, {1, 1, 7, 11, 19, 0, 0, 0} },
        { 5, 11, {1, 1, 5, 1, 1, 0, 0, 0} },
        { 5, 13, {1, 1, 1, 3, 11, 0, 0, 0} },
        { 5, 14, {1, 3, 5, 5, 31, 0, 0, 0} },
        { 6, 1, {1, 3, 3, 9, 7, 49, 0, 0} },
        { 6, 13, {1, 1, 1, 15, 21, 21, 0, 0} },
        { 6, 16, {1, 3, 1, 13, 27, 49, 0, 0} },
        { 6, 19, {1, 1, 1, 15, 7, 5, 0, 0} },
        { 6, 22, {1, 3, 1, 15, 13, 25, 0, 0} },
        { 6, 25, {1, 1, 5, 5, 19, 61, 0, 0} },
        { 7, 1, {1, 3, 7, 11, 23, 15, 103, 0} },
        { 7, 4, {1, 3, 7, 13, 13, 15, 69, 0} },
        { 7, 7, {1, 1, 3, 13, 7, 35, 63, 0} },
        { 7, 8, {1, 3, 5, 9, 1, 25, 53, 0} },
        { 7, 14, {1, 3, 1, 13, 9, 35, 107, 0} },
        { 7, 19, {1, 3, 1, 5, 27, 61, 31, 0} },
        { 7, 21, {1, 1, 5, 11, 19, 41, 61, 0} },
        { 7, 28, {1, 3, 5, 3, 3, 13, 69, 0} },
        { 7, 31, {1, 1, 7, 13, 1, 19, 1, 0} },
        { 7, 32, {1, 3, 3, 5, 5, 19, 33, 0} },
        { 7, 37, {1, 1, 1, 1, 19, 33, 7, 0} },
        { 7, 41, {1, 1, 3, 3, 3, 31, 79, 0} },
        { 7, 42, {1, 3, 1, 13, 13, 27, 95, 0} },
        { 7, 50, {1, 1, 5, 15, 31, 63, 17, 0} }
    };

    //~ V[dim][bit] = 32*32 = 1 kb
    std::array<std::array<std::uint32_t, FOX_SOBOL_BITS>,
               FOX_SOBOL_VERBATIM_DIMS> g_V;
    std::once_flag g_V_built;

    //~ Build the V table
    void build_direction_numbers()
    {
        for (int k = 0; k < FOX_SOBOL_BITS; ++k)
            g_V[0][k] = 1u << (FOX_SOBOL_BITS - 1 - k);

        for (int j = 1; j < FOX_SOBOL_VERBATIM_DIMS; ++j)
        {
            const auto&[s, a, m] = foxJoeKuo[j];

            //~ pack the small odd number into the top
            for (int k = 0; k < s; ++k)
                g_V[j][k] = m[k] << (FOX_SOBOL_BITS - 1 - k);

            //~ v[k] = v[k-s] xor (v[k-s] >> s)
            //~ bit (s-1-i) of a = a_i
            for (int k = s; k < FOX_SOBOL_BITS; ++k)
            {
                std::uint32_t v = g_V[j][k - s] ^ (g_V[j][k - s] >> s);

                for (int i = 1; i < s; ++i)
                {
                    if ((a >> (s - 1 - i)) & 1u)
                        v ^= g_V[j][k - i];
                }
                g_V[j][k] = v;
            }
        }
    }

    //~ Burley hash based Owen scramble
    std::uint32_t owen_scramble(std::uint32_t v,
                                const std::uint32_t seed) noexcept
    {
        v = v ^ (v * 0x3d20adeau);
        v = v + seed;
        v = v * (seed * 2u + 1u);
        v = v ^ (v * 0x05526c56u);
        v = v ^ (v * 0x53a22864u);
        return v;
    }

    //~ 32 bit avalanche hash
    std::uint32_t mix32(std::uint32_t x) noexcept
    {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    //~ [0, 1) * 2^-32
    float u32_to_unit_float(const std::uint32_t x) noexcept
    {
        return std::min(static_cast<float>(x) * 0x1.0p-32f,
                        0x1.fffffep-1f);
    }
} // namespace

fox_tracer::sampling::sobol_sampler::sobol_sampler(
    const bool scrambling, const int max_dimensions,
    const unsigned int seed)
    : scrambling_    (scrambling),
      max_dimensions_(std::max(1, max_dimensions)),
      seed_          (seed)
{
    std::call_once(g_V_built, build_direction_numbers);
}

float fox_tracer::sampling::sobol_sampler::next()
{
    const int d = dim_ % max_dimensions_;
    ++dim_;

    std::uint32_t v = 0;

    if (d < FOX_SOBOL_VERBATIM_DIMS)
    {
        //~ v = XOR of V[d][k] for every bit k of i that is 1
        //~ so we walk the bits of i, low to high and XOR in
        //~ test: i = 0b1011 (=11)
        //~ k=0 bit=1 = v ^= V[d][0]
        //~ k=1 bit=1 = v ^= V[d][1]
        //~ k=2 bit=0 = skip
        //~ k=3 bit=1 = v ^= V[d][3]
        std::uint64_t i = sample_index_;
        int           k = 0;
        while (i > 0 && k < FOX_SOBOL_BITS)
        {
            if (i & 1ull) v ^= g_V[d][k];
            i >>= 1;
            ++k;
        }
    }
    else
    {
        //~ should work
        //~ << 16 | >> 16 swaps the two halves
        //~ then 0x00ff00ff mask swaps bytes within each half
        //~ then 0x0f0f0f0f swaps nibbles within each byte
        //~ then 0x33333333 swaps 2-bit pairs within each nibble
        //~ then 0x55555555 swaps single bits within each pair
        //~ end
        std::uint32_t i32 = static_cast<std::uint32_t>(sample_index_);
        i32 = (i32 << 16) | (i32 >> 16);
        i32 = ((i32 & 0x00ff00ffu) << 8) | ((i32 & 0xff00ff00u) >> 8);
        i32 = ((i32 & 0x0f0f0f0fu) << 4) | ((i32 & 0xf0f0f0f0u) >> 4);
        i32 = ((i32 & 0x33333333u) << 2) | ((i32 & 0xccccccccu) >> 2);
        i32 = ((i32 & 0x55555555u) << 1) | ((i32 & 0xaaaaaaaau) >> 1);
        v = i32;
    }

    if (scrambling_)
    {
        const std::uint32_t dim_seed = mix32(static_cast<std::uint32_t>(d)
                                           ^ (seed_ * 0x9e3779b9u));
        v = owen_scramble(v, dim_seed);
    }

    return u32_to_unit_float(v);
}

void fox_tracer::sampling::sobol_sampler::start_pixel(
    const int px, const int py,
    const int sample_index)
{
    const auto left  = static_cast<std::uint64_t>(static_cast<std::uint32_t>(px)) << 32;
    const auto right = static_cast<std::uint64_t>(static_cast<std::uint32_t>(py));
    const auto pix   = left ^ right;
    sample_index_    = pix * 1009ull + static_cast<std::uint64_t>(sample_index);
    dim_             = 0;
}

void fox_tracer::sampling::sobol_sampler::start_dimension(const int dim)
{
    dim_ = std::clamp(dim, 0, max_dimensions_ - 1);
}

void fox_tracer::sampling::sobol_sampler::reset_with_seed(const unsigned int seed)
{
    seed_         = seed;
    sample_index_ = 0;
    dim_          = 0;
}
