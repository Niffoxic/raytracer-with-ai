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
#ifndef RAYTRACER_WITH_AI_IMAGING_H
#define RAYTRACER_WITH_AI_IMAGING_H

#include "core.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace fox_tracer
{
    constexpr float fox_texel_scale = 1.0f / 256.0f;

    class texture
    {
    public:
        color* texels{nullptr};
        float* alpha {nullptr};
        int    width   {0};
        int    height  {0};
        int    channels{0};

         texture() noexcept = default;
        ~texture();

        texture(const texture&)            = delete;
        texture& operator=(const texture&) = delete;

        texture           (texture&& other) noexcept;
        texture& operator=(texture&& other) noexcept;

        void load_default();
        void load(const std::string& filename);

        [[nodiscard]] color sample      (float tu, float tv) const;
        [[nodiscard]] float sample_alpha(float tu, float tv) const;
    };

    namespace filter
    {
        struct filter_sample
        {
            float x     {0.0f};
            float y     {0.0f};
            float weight{1.0f};
        };

        class image_filter
        {
        public:
            virtual ~image_filter() = default;

            [[nodiscard]] virtual float filter(float x, float y) const = 0;
            [[nodiscard]] virtual int   size  () const = 0;

            [[nodiscard]] virtual vec2 radius_2d() const
            {
                const auto s = static_cast<float>(size());
                return {s, s};
            }

            [[nodiscard]] virtual float evaluate(const float x, const float y) const
            {
                return filter(x, y);
            }

            [[nodiscard]] virtual float         integral() const = 0;
            [[nodiscard]] virtual filter_sample sample  (float u1, float u2) const = 0;
        };

        class filter_sampler
        {
        public:
            explicit filter_sampler(const image_filter& f, int bin_count = 32);
            [[nodiscard]] filter_sample sample(float u1, float u2) const;

        private:
            std::vector<float>              marginal_cdf_;
            std::vector<std::vector<float>> conditional_cdf_;
            std::vector<float>              values_;
            vec2                            radius_;
            int                             bins_;
            float                           integral_abs_;
        };

        class box_filter : public image_filter
        {
        public:
            float rx{0.5f};
            float ry{0.5f};

            explicit box_filter(float _rx = 0.5f, float _ry = 0.5f) noexcept;

            [[nodiscard]] float         filter  (float x, float y)   const override;
            [[nodiscard]] float         evaluate(float x, float y)   const override;
            [[nodiscard]] filter_sample sample  (float u1, float u2) const override;

            [[nodiscard]] vec2  radius_2d() const override;
            [[nodiscard]] float integral () const override;
            [[nodiscard]] int   size     () const override;
        };

 
    }

    struct tonemap_params
    {
        float exposure   {1.0f};
        float gamma      {2.2f};
        float contrast   {1.0f};
        float saturation {1.0f};
    };

    color apply_tonemap(const color& linear_hdr, const tonemap_params& p) noexcept;

    class film
    {
        using filter_type = filter::image_filter;
    public:
        color*           film_buffer{nullptr};
        unsigned int     width {0};
        unsigned int     height{0};
        std::atomic<int> SPP{0};

        std::unique_ptr<filter_type> filter;

        film() noexcept = default;
        ~film();

        film(const film&)            = delete;
        film& operator=(const film&) = delete;
        film(film&&)                 = delete;
        film& operator=(film&&)      = delete;

        void init(int _width, int _height, std::unique_ptr<filter_type> _filter);

        void clear();
        void increment_spp();

        void set_filter(std::unique_ptr<filter_type> new_filter) noexcept;

        void splat(float x, float y, const color& L) const;

        void splat_into(color* buf, int buf_w, int buf_h,
                        int buf_x0, int buf_y0,
                        float x, float y, const color& L) const;

        void splat_importance(color* buf, int buf_w, int buf_h,
                              int buf_x0, int buf_y0,
                              float x, float y, const color& L,
                              float u1, float u2) const;

        void tonemap(int x, int y,
                     unsigned char& r, unsigned char& g, unsigned char& b,
                     const tonemap_params& params) const;

        void save(const std::string& filename) const;
    };

    class adaptive_sampler
    {
    public:
        int img_width   { 0 };
        int img_height  { 0 };
        int block_size  { 16};
        int num_blocks_x{ 0 };
        int num_blocks_y{ 0 };

        std::vector<float> variance;
        std::vector<float> weight;
        std::vector<int>   block_spp;
        std::vector<int>   allocated;

        void init(int width, int height, int _block_size = 16);

        [[nodiscard]]
        int  block_index_for  ( int px, int py)   const noexcept;
        void block_pixel_range( int b, int& x0, int& y0,
                                int& x1, int& y1) const noexcept;

        void compute_variance(const film& f);
        void allocate_samples(int total_samples, int min_per_block = 1);

        [[nodiscard]] int  samples_for_block   (int b)  const noexcept;
                      void record_block_samples(int b, int n) noexcept;
        [[nodiscard]] int  block_spp_of        (int b)  const noexcept;
    };

} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_IMAGING_H
