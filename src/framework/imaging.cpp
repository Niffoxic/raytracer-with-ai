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
#include "framework/imaging.h"
#include "utils/logger.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <utility>

fox_tracer::texture::~texture()
{
    delete[] texels;
    delete[] alpha;
}

fox_tracer::texture::texture(texture &&other) noexcept
    : texels  (other.texels),
      alpha   (other.alpha),
      width   (other.width),
      height  (other.height),
      channels(other.channels)
{
    other.texels   = nullptr;
    other.alpha    = nullptr;
    other.width    = 0;
    other.height   = 0;
    other.channels = 0;
}

fox_tracer::texture& fox_tracer::texture::operator=(texture &&other) noexcept
{
    if (this != &other)
    {
        delete[] texels;
        delete[] alpha;

        texels   = other.texels;
        alpha    = other.alpha;
        width    = other.width;
        height   = other.height;
        channels = other.channels;

        other.texels   = nullptr;
        other.alpha    = nullptr;
        other.width    = 0;
        other.height   = 0;
        other.channels = 0;
    }
    return *this;
}

void fox_tracer::texture::load_default()
{
    delete[] texels;
    delete[] alpha;
    alpha    = nullptr;

    width    = 1;
    height   = 1;

    channels  = 3;
    texels    = new color[1];
    texels[0] = color(1.0f, 1.0f, 1.0f);
}

void fox_tracer::texture::load(const std::string &filename)
{
 delete[] texels;
        delete[] alpha;
        texels = nullptr;
        alpha  = nullptr;
        width  = 0;
        height = 0;

        if (filename.find(".hdr") != std::string::npos)
        {
            float* texture_data = stbi_loadf(filename.c_str(), &width, &height, &channels, 0);

            if (texture_data == nullptr || width <= 0 || height <= 0)
            {
                const char* reason = stbi_failure_reason();
                LOG_WARN("texture") << "stbi_loadf failed: " << filename
                                    << " (" << (reason ? reason : "unknown") << ")";
                load_default();
                return;
            }
            texels = new color[static_cast<size_t>(width) * height];

            for (int i = 0; i < (width * height); ++i)
            {
                texels[i] = color(texture_data[i * channels],
                                  texture_data[(i * channels) + 1],
                                  texture_data[(i * channels) + 2]);
            }
            stbi_image_free(texture_data);
            return;
        }

        unsigned char* texture_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
        if (texture_data == nullptr || width <= 0 || height <= 0)
        {
            const char* reason = stbi_failure_reason();
            LOG_WARN("texture") << "stbi_load failed: " << filename
                                << " (" << (reason ? reason : "unknown") << ")";
            load_default();
            return;
        }

        texels = new color[static_cast<size_t>(width) * height];

        for (int i = 0; i < (width * height); ++i)
        {
            texels[i] = color(
                static_cast<float>(texture_data[i * channels])/ 255.0f,
                static_cast<float>(texture_data[(i * channels) + 1]) / 255.0f,
                static_cast<float>(texture_data[(i * channels) + 2]) / 255.0f);
        }

        if (channels == 4)
        {
            alpha = new float[static_cast<size_t>(width) * height];
            for (int i = 0; i < (width * height); ++i)
            {
                alpha[i] = static_cast<float>(texture_data[(i * channels) + 3]) / 255.0f;
            }
        }
        stbi_image_free(texture_data);
}

fox_tracer::color fox_tracer::texture::sample(const float tu, const float tv) const
{
    if (texels == nullptr || width <= 0 || height <= 0)
    {
        return color(1.0f, 1.0f, 1.0f);
    }
    const float su = std::isfinite(tu) ? tu - std::floor(tu) : 0.0f;
    const float sv = std::isfinite(tv) ? tv - std::floor(tv) : 0.0f;

    const float u = su * static_cast<float>(width);
    const float v = sv * static_cast<float>(height);

    int x = static_cast<int>(std::floor(u));
    int y = static_cast<int>(std::floor(v));

    const float frac_u = u - static_cast<float>(x);
    const float frac_v = v - static_cast<float>(y);

    const float w0 = (1.0f - frac_u) * (1.0f - frac_v);
    const float w1 = frac_u * (1.0f - frac_v);
    const float w2 = (1.0f - frac_u) * frac_v;
    const float w3 = frac_u * frac_v;

    x = std::clamp(x, 0, width  - 1);
    y = std::clamp(y, 0, height - 1);

    const int x1 = (x + 1) % width;
    const int y1 = (y + 1) % height;

    color s[4];
    s[0] = texels[y  * width + x ];
    s[1] = texels[y  * width + x1];
    s[2] = texels[y1 * width + x ];
    s[3] = texels[y1 * width + x1];

    return (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
}

float fox_tracer::texture::sample_alpha(const float tu, const float tv) const
{
    if (alpha == nullptr || width <= 0 || height <= 0)
    {
        return 1.0f;
    }

    const float su = std::isfinite(tu) ? tu - std::floor(tu) : 0.0f;
    const float sv = std::isfinite(tv) ? tv - std::floor(tv) : 0.0f;

    const float u = su * static_cast<float>(width);
    const float v = sv * static_cast<float>(height);

    int x = static_cast<int>(std::floor(u));
    int y = static_cast<int>(std::floor(v));

    const float frac_u = u - static_cast<float>(x);
    const float frac_v = v - static_cast<float>(y);

    const float w0 = (1.0f - frac_u) * (1.0f - frac_v);
    const float w1 = frac_u * (1.0f - frac_v);
    const float w2 = (1.0f - frac_u) * frac_v;
    const float w3 = frac_u * frac_v;

    x = std::clamp(x, 0, width  - 1);
    y = std::clamp(y, 0, height - 1);

    const int x1 = (x + 1) % width;
    const int y1 = (y + 1) % height;

    float s[4];
    s[0] = alpha[y  * width + x ];
    s[1] = alpha[y  * width + x1];
    s[2] = alpha[y1 * width + x ];
    s[3] = alpha[y1 * width + x1];

    return (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
}

fox_tracer::filter::filter_sampler::filter_sampler(const image_filter &f, const int bin_count)
    : radius_(f.radius_2d()), bins_(bin_count), integral_abs_(0.f)
{
    if (bins_ < 1) bins_ = 1;

    values_.assign(static_cast<size_t>(bins_) * bins_, 0.0f);
    conditional_cdf_.assign(bins_, std::vector<float>(bins_ + 1, 0.0f));
    marginal_cdf_.assign(bins_ + 1, 0.0f);

    for (int j = 0; j < bins_; ++j)
    {
        const float v = (static_cast<float>(j) + 0.5f) / static_cast<float>(bins_);
        const float y = (2.0f * v - 1.0f) * radius_.y;

        float row_sum = 0.0f;
        conditional_cdf_[j][0] = 0.0f;
        for (int i = 0; i < bins_; ++i)
        {
            const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(bins_);
            const float x = (2.0f * u - 1.0f) * radius_.x;
            const float val = std::fabs(f.evaluate(x, y));
            values_[static_cast<size_t>(j) * bins_ + i] = val;
            row_sum += val;
            conditional_cdf_[j][i + 1] = row_sum;
        }
        if (row_sum > 0.0f)
        {
            const float inv = 1.0f / row_sum;
            for (int i = 1; i <= bins_; ++i) conditional_cdf_[j][i] *= inv;
        }
        marginal_cdf_[j + 1] = marginal_cdf_[j] + row_sum;
    }
    integral_abs_ = marginal_cdf_[bins_];
    if (integral_abs_ > 0.0f)
    {
        const float inv = 1.0f / integral_abs_;
        for (int j = 1; j <= bins_; ++j) marginal_cdf_[j] *= inv;
    }
}

fox_tracer::filter::filter_sample fox_tracer::filter::filter_sampler::sample(
    const float u1, const float u2) const
{
    if (integral_abs_ <= 0.0f)
    {
        // Degenerate filter
        return { (2.0f * u1 - 1.0f) * radius_.x,
                 (2.0f * u2 - 1.0f) * radius_.y,
                 0.0f };
    }

    // Binary search - CDF for the y bin
    auto search_cdf = [](const std::vector<float>& cdf, const float u, const int n)
    {
        int lo = 0, hi = n;
        while (lo < hi)
        {
            const int mid = (lo + hi) >> 1;
            if (cdf[mid + 1] <= u)
                lo = mid + 1;
            else hi = mid;
        }
        return std::min(lo, n - 1);
    };

    const int j = search_cdf(marginal_cdf_, u2, bins_);
    const float y0  = marginal_cdf_[j];
    const float y1  = marginal_cdf_[j + 1];
    const float dy  = (y1 > y0) ? (u2 - y0) / (y1 - y0) : 0.5f;
    const float yv  = (static_cast<float>(j) + dy) / static_cast<float>(bins_);
    const float y   = (2.0f * yv - 1.0f) * radius_.y;

    const int i = search_cdf(conditional_cdf_[j], u1, bins_);
    const float x0  = conditional_cdf_[j][i];
    const float x1  = conditional_cdf_[j][i + 1];
    const float dx  = (x1 > x0) ? (u1 - x0) / (x1 - x0) : 0.5f;
    const float xv  = (static_cast<float>(i) + dx) / static_cast<float>(bins_);
    const float x   = (2.0f * xv - 1.0f) * radius_.x;

    return { x, y, 1.0f };
}

fox_tracer::filter::box_filter::box_filter(const float _rx, const float _ry) noexcept
    : rx(_rx), ry(_ry)
{}

float fox_tracer::filter::box_filter::filter(const float x, const float y) const
{
    return evaluate(x, y);
}

float fox_tracer::filter::box_filter::evaluate(const float x, const float y) const
{
    if (std::fabs(x) <= rx && std::fabs(y) <= ry)
    {
        return 1.0f;
    }
    return 0.0f;
}

fox_tracer::filter::filter_sample fox_tracer::filter::box_filter::sample(
    const float u1, const float u2) const
{
    return { (2.0f * u1 - 1.0f) * rx,
             (2.0f * u2 - 1.0f) * ry,
             1.0f };
}

fox_tracer::vec2 fox_tracer::filter::box_filter::radius_2d() const
{
    return {rx, ry};;
}

float fox_tracer::filter::box_filter::integral() const
{
    return 4.0f * rx * ry;
}

int fox_tracer::filter::box_filter::size() const
{
    if (rx <= 0.5f && ry <= 0.5f) return 0;

    return std::max(static_cast<int>(std::ceil(rx)),
                    static_cast<int>(std::ceil(ry)));
}

fox_tracer::filter::gaussian_filter::gaussian_filter(
    const float rx, const float ry, const float _alpha)
    : radius_xy(rx, ry), alpha(_alpha)
{
    exp_rx = std::exp(-alpha * rx * rx);
    exp_ry = std::exp(-alpha * ry * ry);
}

float fox_tracer::filter::gaussian_filter::gaussian_1d(
    const float d, const float exp_at_radius) const
{
    const float v = std::exp(-alpha * d * d) - exp_at_radius;
    return v > 0.0f ? v : 0.0f;
}

float fox_tracer::filter::gaussian_filter::filter(const float x, const float y) const
{
    return evaluate(x, y);
}

float fox_tracer::filter::gaussian_filter::evaluate(const float x, const float y) const
{
    if (std::fabs(x) > radius_xy.x || std::fabs(y) > radius_xy.y) return 0.0f;
    return gaussian_1d(x, exp_rx) * gaussian_1d(y, exp_ry);
}

fox_tracer::filter::filter_sample fox_tracer::filter::gaussian_filter::sample(const float u1, const float u2) const
{
    const float sqrt_a = std::sqrt(alpha);

    auto sample_axis = [&](const float u, const float r)
    {
        const float arg     = u * std::erf(r * sqrt_a);
        const float clamped = std::clamp(arg, -0.9999999f, 0.9999999f);
        return math::erf_inv_approx<float>(clamped) / sqrt_a;
    };

    const float ux = 2.0f * u1 - 1.0f;
    const float uy = 2.0f * u2 - 1.0f;

    return { sample_axis(ux, radius_xy.x),
             sample_axis(uy, radius_xy.y),
             1.0f };
}

fox_tracer::vec2 fox_tracer::filter::gaussian_filter::radius_2d() const
{
    return radius_xy;
}

float fox_tracer::filter::gaussian_filter::integral() const
{
    const float sqrt_a       = std::sqrt(alpha);
    const float sqrt_pi_over_a = std::sqrt(math::pi<float> / alpha);

    const float ix = sqrt_pi_over_a * std::erf(radius_xy.x * sqrt_a)
                   - 2.0f * radius_xy.x * exp_rx;
    const float iy = sqrt_pi_over_a * std::erf(radius_xy.y * sqrt_a)
                   - 2.0f * radius_xy.y * exp_ry;

    return ix * iy;
}

int fox_tracer::filter::gaussian_filter::size() const
{
    return std::max(static_cast<int>(std::ceil(radius_xy.x)),
                    static_cast<int>(std::ceil(radius_xy.y)));
}

fox_tracer::filter::mitchell_netravali_filter::mitchell_netravali_filter(
    const float _b, const float _c,
    const float rx, const float ry)
: b_param(_b), c_param(_c), radius_xy(rx, ry)
{
    sampler_ = std::make_unique<filter_sampler>(*this, 32);
}

float fox_tracer::filter::mitchell_netravali_filter::mitchell_1d(float x) const
{
    //~ TODO: make B, C configurable from imgui so that I can have B-spile fam
    x = std::fabs(x);
    if (x < 1.0f)
    {
        return (1.0f / 6.0f) * (
            (12.0f - 9.0f * b_param - 6.0f * c_param) * x * x * x +
            (-18.0f + 12.0f * b_param + 6.0f * c_param) * x * x +
            (6.0f - 2.0f * b_param));
    }
    if (x < 2.0f)
    {
        return (1.0f / 6.0f) * (
            (-b_param - 6.0f * c_param) * x * x * x +
            (6.0f * b_param + 30.0f * c_param) * x * x +
            (-12.0f * b_param - 48.0f * c_param) * x +
            (8.0f * b_param + 24.0f * c_param));
    }
    return 0.0f;
}

float fox_tracer::filter::mitchell_netravali_filter::filter(const float x, const float y) const
{
    const float sx = (radius_xy.x > 0.0f) ? (2.0f / radius_xy.x) : 0.0f;
    const float sy = (radius_xy.y > 0.0f) ? (2.0f / radius_xy.y) : 0.0f;
    return mitchell_1d(x * sx) * mitchell_1d(y * sy);
}

float fox_tracer::filter::mitchell_netravali_filter::evaluate(const float x, const float y) const
{
        return filter(x, y);
}

fox_tracer::filter::filter_sample fox_tracer::filter::mitchell_netravali_filter::sample(
    const float u1, const float u2) const
{
    filter_sample s = sampler_->sample(u1, u2);
    if (evaluate(s.x, s.y) < 0.0f) s.weight = -s.weight;
    return s;
}

fox_tracer::vec2 fox_tracer::filter::mitchell_netravali_filter::radius_2d() const
{
    return radius_xy;
}

float fox_tracer::filter::mitchell_netravali_filter::integral() const
{
    return (radius_xy.x * 0.5f) * (radius_xy.y * 0.5f);
}

int fox_tracer::filter::mitchell_netravali_filter::size() const
{
    return std::max(static_cast<int>(std::ceil(radius_xy.x)),
                    static_cast<int>(std::ceil(radius_xy.y)));
}

fox_tracer::filter::triangle_filter::triangle_filter(const float rx, const float ry) noexcept
    : radius_xy(rx, ry)
{}

float fox_tracer::filter::triangle_filter::filter(const float x, const float y) const
{
    return evaluate(x, y);
}

float fox_tracer::filter::triangle_filter::evaluate(float x, float y) const
{
    const float fx = std::max(0.0f, 1.0f - std::fabs(x) / radius_xy.x);
    const float fy = std::max(0.0f, 1.0f - std::fabs(y) / radius_xy.y);
    return fx * fy;
}

fox_tracer::filter::filter_sample fox_tracer::filter::triangle_filter::sample(
    const float u1, const float u2) const
{
    auto tri = [](const float u)
    {
        return (u < 0.5f) ? (std::sqrt(2.0f * u) - 1.0f)
                          : (1.0f - std::sqrt(2.0f - 2.0f * u));
    };

    return { tri(u1) * radius_xy.x,
             tri(u2) * radius_xy.y,
             1.0f };
}

fox_tracer::vec2 fox_tracer::filter::triangle_filter::radius_2d() const
{
    return radius_xy;
}

float fox_tracer::filter::triangle_filter::integral() const
{
    return radius_xy.x * radius_xy.y;
}

int fox_tracer::filter::triangle_filter::size() const
{
    return std::max(static_cast<int>(std::ceil(radius_xy.x)),
                static_cast<int>(std::ceil(radius_xy.y)));
}

fox_tracer::filter::lanczos_sinc_filter::lanczos_sinc_filter(float rx, float ry, float _tau)
    : radius_xy(rx, ry), tau(_tau)
{
    sampler_ = std::make_unique<filter_sampler>(*this, 32);
}

float fox_tracer::filter::lanczos_sinc_filter::filter(float x, float y) const
{
    return evaluate(x, y);
}

float fox_tracer::filter::lanczos_sinc_filter::evaluate(float x, float y) const
{
    return math::windowed_sinc<float>(x, radius_xy.x, tau)
         * math::windowed_sinc<float>(y, radius_xy.y, tau);
}

fox_tracer::filter::filter_sample fox_tracer::filter::lanczos_sinc_filter::sample(float u1, float u2) const
{
    filter_sample s = sampler_->sample(u1, u2);
    if (evaluate(s.x, s.y) < 0.0f) s.weight = -s.weight;
    return s;
}

fox_tracer::vec2 fox_tracer::filter::lanczos_sinc_filter::radius_2d() const
{
    return radius_xy;
}

float fox_tracer::filter::lanczos_sinc_filter::integral() const
{
    return 2.0f * radius_xy.x * 2.0f * radius_xy.y;
}

int fox_tracer::filter::lanczos_sinc_filter::size() const
{
    return std::max(static_cast<int>(std::ceil(radius_xy.x)),
                    static_cast<int>(std::ceil(radius_xy.y)));
}

// TODO: look at some famous tonemap from some games and create an enum for selecting any of them runtime
namespace
{
    inline fox_tracer::color tm_reinhard(const fox_tracer::color& c) noexcept
    {
        return fox_tracer::color(c.red   / (1.0f + c.red),
                                 c.green / (1.0f + c.green),
                                 c.blue  / (1.0f + c.blue));
    }

    inline fox_tracer::color apply_contrast_saturation(
        const fox_tracer::color& c,
        const float contrast,
        const float saturation) noexcept
    {
        const float lum = c.luminance();
        auto out = fox_tracer::color(
             lum + (c.red - lum)   * saturation,
            lum + (c.green - lum) * saturation,
            lum + (c.blue - lum)  * saturation);

        out = fox_tracer::color( (out.red   - 0.5f) * contrast + 0.5f,
                                (out.green - 0.5f) * contrast + 0.5f,
                                (out.blue  - 0.5f) * contrast + 0.5f);
        return out;
    }
}

fox_tracer::color fox_tracer::apply_tonemap(const color &linear_hdr, const tonemap_params &p) noexcept
{
    const color exposed = linear_hdr * p.exposure;
    color mapped        = tm_reinhard(exposed);
    mapped              = apply_contrast_saturation(mapped, p.contrast, p.saturation);
    return math::apply_gamma(mapped, p.gamma);
}

fox_tracer::film::~film()
{
    delete[] film_buffer;
}

void fox_tracer::film::init(const int _width, const int _height, std::unique_ptr<filter_type> _filter)
{
    width  = static_cast<unsigned int>(_width);
    height = static_cast<unsigned int>(_height);
    delete[] film_buffer;

    film_buffer = new color[static_cast<size_t>(width) * height];
    clear();
    filter = std::move(_filter);
}

void fox_tracer::film::clear()
{
    const size_t n = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < n; ++i)
    {
        film_buffer[i] = color();
    }
    SPP.store(0, std::memory_order_relaxed);
}

void fox_tracer::film::increment_spp()
{
    SPP.fetch_add(1, std::memory_order_relaxed);
}

void fox_tracer::film::set_filter(std::unique_ptr<filter_type> new_filter) noexcept
{
    filter = std::move(new_filter);
}

void fox_tracer::film::splat(const float x, const float y, const color &L) const
{
    const int s = filter->size();
    const int span = 2 * s + 1;
    const int cap  = span * span;

    std::vector<float>        filter_weights(cap);
    std::vector<unsigned int> indices       (cap);
    unsigned int used  = 0;
    float        total = 0.0f;

    for (int i = -s; i <= s; ++i)
    {
        for (int j = -s; j <= s; ++j)
        {
            const int px = static_cast<int>(x) + j;
            const int py = static_cast<int>(y) + i;

            if (px >= 0 && px < static_cast<int>(width) &&
                py >= 0 && py < static_cast<int>(height))
            {
                indices       [used] = py * width + px;
                filter_weights[used] = filter->filter(
                    static_cast<float>(px) - x,
                    static_cast<float>(py) - y);
                total += filter_weights[used];
                ++used;
            }
        }
    }

    if (total <= 0.0f) return;

    for (unsigned int i = 0; i < used; ++i)
    {
        film_buffer[indices[i]] = film_buffer[indices[i]]
                                + (L * (filter_weights[i] / total));
    }
}

void fox_tracer::film::splat_into(
    color *buf, const int buf_w, const int buf_h,
    const int buf_x0, const int buf_y0,
    const float x, const float y,
    const color &L) const
{
    const int s = filter->size();
    if (s == 0)
    {
        const int px = static_cast<int>(x);
        const int py = static_cast<int>(y);

        if (px < 0 || px >= static_cast<int>(width) ||
            py < 0 || py >= static_cast<int>(height)) return;

        const int lx = px - buf_x0;
        const int ly = py - buf_y0;

        if (lx < 0 || lx >= buf_w || ly < 0 || ly >= buf_h) return;

        buf[ly * buf_w + lx] = buf[ly * buf_w + lx] + L;
        return;
    }

    const int span = 2 * s + 1;
    const int cap  = span * span;

    std::vector<float> filter_weights(cap);
    std::vector<int>   buf_idx       (cap);
    int   used  = 0;
    float total = 0.0f;

    const int base_x = static_cast<int>(x);
    const int base_y = static_cast<int>(y);
    const int w_int  = static_cast<int>(width);
    const int h_int  = static_cast<int>(height);

    for (int i = -s; i <= s; ++i)
    {
        const int py = base_y + i;
        if (py < 0 || py >= h_int) continue;

        const int ly = py - buf_y0;
        if (ly < 0 || ly >= buf_h) continue;

        const float dy = static_cast<float>(py) - y;

        for (int j = -s; j <= s; ++j)
        {
            const int px = base_x + j;
            if (px < 0 || px >= w_int) continue;

            const int lx = px - buf_x0;
            if (lx < 0 || lx >= buf_w) continue;

            const float w = filter->filter(
                static_cast<float>(px) - x, dy);

            if (w == 0.0f) continue;

            buf_idx       [used] = ly * buf_w + lx;
            filter_weights[used] = w;
            total += w;
            ++used;
        }
    }
    if (total <= 0.0f) return;

    const float inv_total = 1.0f / total;

    for (int i = 0; i < used; ++i)
    {
        buf[buf_idx[i]] = buf[buf_idx[i]]
                        + (L * (filter_weights[i] * inv_total));
    }
}

void fox_tracer::film::splat_importance(
    color *buf, const int buf_w, const int buf_h,
    const int buf_x0, const int buf_y0, float x, const float y,
    const color &L, const float u1, const float u2) const
{
    if (filter == nullptr) return;

    const auto fs = filter->sample(u1, u2);

    const float deposit_x = x + fs.x;
    const float deposit_y = y + fs.y;

    const int   px = static_cast<int>(deposit_x);
    const int   py = static_cast<int>(deposit_y);

    if (px < 0 || px >= static_cast<int>(width) ||
        py < 0 || py >= static_cast<int>(height)) return;

    const int lx = px - buf_x0;
    const int ly = py - buf_y0;

    if (lx < 0 || lx >= buf_w || ly < 0 || ly >= buf_h) return;

    const color contribution = L * fs.weight;
    buf[ly * buf_w + lx] = buf[ly * buf_w + lx] + contribution;
}

void fox_tracer::film::tonemap(
    const int x, const int y, unsigned char &r,
    unsigned char &g, unsigned char &b,
    const tonemap_params &params) const
{
    const int spp_now = SPP.load(std::memory_order_relaxed);
    const int spp     = spp_now > 0 ? spp_now : 1;
    const color pixel = film_buffer[y * width + x] / static_cast<float>(spp);

    const color mapped = apply_tonemap(pixel, params);

    r = static_cast<unsigned char>(math::saturate(mapped.red)   * 255.0f);
    g = static_cast<unsigned char>(math::saturate(mapped.green) * 255.0f);
    b = static_cast<unsigned char>(math::saturate(mapped.blue)  * 255.0f);
}

void fox_tracer::film::save(const std::string &filename) const
{
    const size_t n = static_cast<size_t>(width) * height;
    auto* hdr_pixels = new color[n];

    const int spp = std::max(1, SPP.load(std::memory_order_relaxed));

    for (size_t i = 0; i < n; ++i)
    {
        hdr_pixels[i] = film_buffer[i] / static_cast<float>(spp);
    }

    stbi_write_hdr(filename.c_str(),
                   static_cast<int>(width), static_cast<int>(height),
                   3, reinterpret_cast<float*>(hdr_pixels));

    delete[] hdr_pixels;
}

void fox_tracer::adaptive_sampler::init(const int width, const int height, const int _block_size)
{
    // TODO: quad-tree subdivide blocks with high variance instead of fixed grid
    img_width    = width;
    img_height   = height;

    block_size   = std::max(1, _block_size);
    num_blocks_x = (img_width  + block_size - 1) / block_size;
    num_blocks_y = (img_height + block_size - 1) / block_size;

    const int n = num_blocks_x * num_blocks_y;
    variance .assign(n, 0.0f);
    weight   .assign(n, 1.0f / static_cast<float>(std::max(1, n)));
    block_spp.assign(n, 0);
    allocated.assign(n, 1);
}

int fox_tracer::adaptive_sampler::block_index_for(const int px, const int py) const noexcept
{
    const int bx = std::min(num_blocks_x - 1, px / block_size);
    const int by = std::min(num_blocks_y - 1, py / block_size);

    return by * num_blocks_x + bx;
}

void fox_tracer::adaptive_sampler::block_pixel_range(
    int b, int &x0,
    int &y0, int &x1, int &y1) const noexcept
{
    const int bx = b % num_blocks_x;
    const int by = b / num_blocks_x;

    x0 = bx * block_size;
    y0 = by * block_size;

    x1 = std::min(img_width,  x0 + block_size);
    y1 = std::min(img_height, y0 + block_size);
}

void fox_tracer::adaptive_sampler::compute_variance(const film &f)
{
    // TODO: relative variance  so dim regions dont get starved
    // TODO: edge aware variance downweight neighbour difference noise

    //~ reference
    //~ per block sample variance of luminance
    //~ mean = (1/N) * sum_i L_i
    //~ var = (1/(N-1)) * sum_i (L_i - mean)^2

    const int n_blocks = num_blocks_x * num_blocks_y;
    const int spp_now  = f.SPP.load(std::memory_order_relaxed);
    const int spp      = (spp_now > 0) ? spp_now : 1;

    for (int b = 0; b < n_blocks; ++b)
    {
        int x0, y0, x1, y1;
        block_pixel_range(b, x0, y0, x1, y1);

        // double mean = 0.0, m2 = 0.0;
        // int    cnt  = 0;
        // for (int y = y0; y < y1; ++y)
        // {
        //     for (int x = x0; x < x1; ++x)
        //     {
        //         const double l = f.film_buffer[y * f.width + x].luminance()
        //                        / static_cast<double>(spp);
        //         ++cnt;
        //         const double delta  = l - mean;
        //         mean += delta / cnt;
        //         m2   += delta * (l - mean);
        //     }
        // }
        // variance[b] = (cnt > 1) ? static_cast<float>(m2 / (cnt - 1)) : 0.0f;

        double sum = 0.0;
        int    cnt = 0;
        for (int y = y0; y < y1; ++y)
        {
            for (int x = x0; x < x1; ++x)
            {
                const color c = f.film_buffer[y * f.width + x] / static_cast<float>(spp);
                sum += static_cast<double>(c.luminance());
                ++cnt;
            }
        }
        if (cnt < 2) { variance[b] = 0.0f; continue; }
        const double mean = sum / static_cast<double>(cnt);
        // variance[b] = static_cast<float>(acc / static_cast<double>(cnt));

        double acc = 0.0;
        for (int y = y0; y < y1; ++y)
        {
            for (int x = x0; x < x1; ++x)
            {
                const color c = f.film_buffer[y * f.width + x] / static_cast<float>(spp);
                const double d = static_cast<double>(c.luminance()) - mean;
                acc += d * d;
            }
        }
        variance[b] = static_cast<float>(acc / static_cast<double>(cnt - 1));
    }
}

void fox_tracer::adaptive_sampler::allocate_samples(const int total_samples, const int min_per_block)
{
    // TODO: clamp max samples per block
    // TODO: per block convergence test

    const int n_blocks = num_blocks_x * num_blocks_y;
    if (n_blocks == 0) return;

    double sum_v = 0.0;
    for (int b = 0; b < n_blocks; ++b) sum_v += static_cast<double>(variance[b]);

    if (sum_v <= 0.0)
    {
        const int even = std::max(min_per_block, total_samples / n_blocks);
        for (int b = 0; b < n_blocks; ++b)
        {
            weight   [b] = 1.0f / static_cast<float>(n_blocks);
            allocated[b] = even;
        }
        return;
    }

    // sum_v = 0.0;
    // for (int b = 0; b < n_blocks; ++b)
    // {
    //     sum_v += std::sqrt(variance[b]);
    //     weight[b] = std::sqrt(variance[b]) / sum_v;
    // }

    // const double T = 1.0;
    // for (int b = 0; b < n_blocks; ++b)
    // {
    //     sum_v += std::exp(variance[b] / T);
    //     weight[b] = std::exp(variance[b] / T) / sum_v;
    // }

    const int reserved  = min_per_block * n_blocks;
    const int remaining = std::max(0, total_samples - reserved);

    int assigned_extra = 0;
    for (int b = 0; b < n_blocks; ++b)
    {
        weight[b] = static_cast<float>(static_cast<double>(variance[b]) / sum_v);

        // const int extra = static_cast<int>(
        //     std::round(static_cast<double>(weight[b]) * static_cast<double>(remaining)));

        const int extra = static_cast<int>(
            std::floor(static_cast<double>(weight[b]) * static_cast<double>(remaining)));

        allocated[b]   = min_per_block + extra;
        assigned_extra += extra;
    }

    int leftover = remaining - assigned_extra;
    if (leftover > 0)
    {
        std::vector<int> idx(n_blocks);
        for (int i = 0; i < n_blocks; ++i) idx[i] = i;

        // const int top_k = std::min(leftover, n_blocks);
        // std::ranges::partial_sort(idx, idx.begin() + top_k,
        //     [this](const int a, const int b) {
        //         return variance[a] > variance[b];
        //     });
        //
        // std::nth_element(idx.begin(), idx.begin() + leftover, idx.end(),
        //     [this](const int a, const int b) {
        //         return variance[a] > variance[b];
        //     });

        std::ranges::sort(idx,
        [this](const int a, const int b)
        {
            return variance[a] > variance[b];
        });

        // const int top_k = std::min(leftover, n_blocks);
        // for (int i = 0; i < top_k; ++i) ++allocated[idx[i]];

        // leftover sample
        int i = 0;
        while (leftover > 0)
        {
            ++allocated[idx[i % n_blocks]];
            ++i;
            --leftover;
        }
    }
}

int fox_tracer::adaptive_sampler::samples_for_block(const int b) const noexcept
{
    return allocated[b];
}

void fox_tracer::adaptive_sampler::record_block_samples(const int b, const int n) noexcept
{
    block_spp[b] += n; // debugging stats
}

int fox_tracer::adaptive_sampler::block_spp_of(const int b) const noexcept
{
    return block_spp[b];
}
