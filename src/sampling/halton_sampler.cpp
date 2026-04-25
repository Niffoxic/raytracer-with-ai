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
#include "sampler/halton_sampler.h"

#include <algorithm>
#include <array>
#include <vector>

namespace //~ Helpers TODO: create a number theory library for such cases
{
    constexpr int FOX_NUM_PRIMES  = 1024;
    constexpr int FOX_SIEVE_UPPER = 8400;

    //~ Sieve of eratosthene algo
    const std::array<int, FOX_NUM_PRIMES>& primes()
    {
        static const std::array<int, FOX_NUM_PRIMES> table = []()
        {
            std::vector<bool> is_prime(FOX_SIEVE_UPPER + 1, false);
            std::array<int, FOX_NUM_PRIMES> result{};
            int count = 0;
            for (int n = 2; n <= FOX_SIEVE_UPPER && count < FOX_NUM_PRIMES; ++n)
            {
                if (is_prime[n]) continue;
                result[count++] = n;
                for (long long m = static_cast<long long>(n) * n; m <= FOX_SIEVE_UPPER; m += n)
                {
                    is_prime[static_cast<std::size_t>(m)] = true;
                }
            }
            return result;
        }();
        return table;
    }

    //~ Van der Corput ri in arbitrary prime base from pbrt
    float radical_inverse(const int base, std::uint64_t i) noexcept
    {
        const float inv_base   = 1.0f / static_cast<float>(base);
        float       inv_base_n = 1.0f;
        std::uint64_t reversed = 0;

        while (i > 0)
        {
            const std::uint64_t next  = i / static_cast<std::uint64_t>(base);
            const std::uint64_t digit = i - next * static_cast<std::uint64_t>(base);
            reversed    = reversed * static_cast<std::uint64_t>(base) + digit;
            inv_base_n *= inv_base;
            i           = next;
        }

        return std::min(static_cast<float>(reversed) * inv_base_n, 0x1.fffffep-1f);
    }

    //~ Faure lemi.. per digit permutation
    std::uint64_t hash_digit(const std::uint64_t digit,
                                const int base,
                                const int digit_index,
                                const unsigned int seed) noexcept
    {
        std::uint64_t h = digit
            + 0x9e3779b97f4a7c15ull * static_cast<std::uint64_t>(base)
            + 0xc2b2ae3d27d4eb4full * static_cast<std::uint64_t>(digit_index)
            + 0x165667b19e3779f9ull * static_cast<std::uint64_t>(seed);
        h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 33;
        return h % static_cast<std::uint64_t>(base);
    }

    //~ hashed digit permutation kills halton stripes
    float scrambled_inverse(const int base, std::uint64_t i,
                            const unsigned int seed) noexcept
    {
        const float inv_base = 1.0f / static_cast<float>(base);
        float        inv_base_n = 1.0f;
        std::uint64_t reversed = 0;
        int           digit_index = 0;
        while (i > 0)
        {
            const std::uint64_t next  = i / static_cast<std::uint64_t>(base);
            const std::uint64_t digit = i - next * static_cast<std::uint64_t>(base);
            const std::uint64_t perm  = hash_digit(digit, base,
                                                   digit_index, seed);
            reversed   = reversed * static_cast<std::uint64_t>(base) + perm;
            inv_base_n *= inv_base;
            i = next;
            ++digit_index;
        }
        return std::min(static_cast<float>(reversed) * inv_base_n,
                        0x1.fffffep-1f);
    }
} // namespace

fox_tracer::sampling::halton_sampler::halton_sampler(
    const bool scrambling,
    const int max_dimensions,
    const unsigned int seed)
    : scrambling_    (scrambling),
      max_dimensions_(std::clamp(max_dimensions, 2, FOX_NUM_PRIMES)),
      seed_          (seed)
{}

float fox_tracer::sampling::halton_sampler::next()
{
    const int d   = dim_ % max_dimensions_;
    const float v = scrambled_radical_inverse(d, sample_index_);
    ++dim_;
    return v;
}

void fox_tracer::sampling::halton_sampler::start_pixel(
    const int px, const int py,
    const int sample_index)
{
    // px = 257, py = 42, si = 7
    // px = (32 free)(32) bits
    // px <<= 16 (16 free)(32)(16 for py)
    // px = (32)(32) bits
    // px ^ py = (px on left) (py on right)
    const auto left = static_cast<uint64_t>(static_cast<uint32_t>(px)) << 16;
    const auto right = static_cast<uint64_t>(static_cast<uint32_t>(py));
    const auto pix_hash = left ^ right;
    sample_index_   = pix_hash * 1009ull + static_cast<std::uint64_t>(sample_index);
    dim_            = 0;
}

void fox_tracer::sampling::halton_sampler::start_dimension(const int dim)
{
    dim_ = std::clamp(dim, 0, max_dimensions_ - 1);
}

void fox_tracer::sampling::halton_sampler::reset_with_seed(const unsigned int seed)
{
    seed_         = seed;
    sample_index_ = 0;
    dim_          = 0;
}

float fox_tracer::sampling::halton_sampler::scrambled_radical_inverse(
    const int dim, const std::uint64_t i) const
{
    const int base = primes()[dim];
    return scrambling_
        ? scrambled_inverse(base, i, seed_)
        : radical_inverse  (base, i);
}
