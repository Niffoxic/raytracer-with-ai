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
#include "framework/core.h"

#include <cmath>
#include <cstring>
#include <smmintrin.h>

namespace fox_tracer
{
    color::color() noexcept
        : red(0.0f), green(0.0f), blue(0.0f)
    {}

    color::color(const float r, const float g, const float b) noexcept
        : red(r), green(g), blue(b)
    {}

    color::color(const unsigned char r,
                 const unsigned char g,
                 const unsigned char b) noexcept
        : red  (static_cast<float>(r) / 255.0f),
          green(static_cast<float>(g) / 255.0f),
          blue (static_cast<float>(b) / 255.0f)
    {}

    void color::to_rgb(unsigned char& r,
                       unsigned char& g,
                       unsigned char& b) const noexcept
    {
        r = static_cast<unsigned char>(math::saturate(red)   * 255.0f);
        g = static_cast<unsigned char>(math::saturate(green) * 255.0f);
        b = static_cast<unsigned char>(math::saturate(blue)  * 255.0f);
    }

    color color::operator+(const color& rhs) const noexcept
    {
        return { red + rhs.red, green + rhs.green, blue + rhs.blue };
    }

    color color::operator-(const color& rhs) const noexcept
    {
        return { red - rhs.red, green - rhs.green, blue - rhs.blue };
    }

    color color::operator*(const color& rhs) const noexcept
    {
        return { red * rhs.red, green * rhs.green, blue * rhs.blue };
    }

    color color::operator/(const color& rhs) const noexcept
    {
        return { red / rhs.red, green / rhs.green, blue / rhs.blue };
    }

    color color::operator*(const float v) const noexcept
    {
        return { red * v, green * v, blue * v };
    }

    color color::operator/(const float v) const noexcept
    {
        return { red / v, green / v, blue / v };
    }

    float color::luminance() const noexcept
    {
        return (0.2126f * red) + (0.7152f * green) + (0.0722f * blue);
    }

    vec3::vec3() noexcept
        : x(0.0f), y(0.0f), z(0.0f), w(1.0f)
    {}

    vec3::vec3(const float _x, const float _y,
               const float _z, const float _w) noexcept
        : x(_x), y(_y), z(_z), w(_w)
    {}

    vec3 vec3::operator+(const vec3& rhs) const noexcept
    {
        return { x + rhs.x, y + rhs.y, z + rhs.z, 1.0f };
    }

    vec3 vec3::operator-(const vec3& rhs) const noexcept
    {
        return { x - rhs.x, y - rhs.y, z - rhs.z, 1.0f };
    }

    vec3 vec3::operator*(const vec3& rhs) const noexcept
    {
        return { x * rhs.x, y * rhs.y, z * rhs.z, 1.0f };
    }

    vec3 vec3::operator*(const float scalar) const noexcept
    {
        return { x * scalar, y * scalar, z * scalar, 1.0f };
    }

    vec3 vec3::operator/(const float scalar) const noexcept
    {
        const float inv = 1.0f / scalar;
        return { x * inv, y * inv, z * inv, 1.0f };
    }

    vec3 vec3::operator-() const noexcept
    {
        return { -x, -y, -z, 1.0f };
    }

    vec3 vec3::perspective_divide() const noexcept
    {
        const float inv_w = 1.0f / w;
        return { x * inv_w, y * inv_w, z * inv_w, inv_w };
    }

    float vec3::length_squared() const noexcept
    {
        return (x * x) + (y * y) + (z * z);
    }

    float vec3::length() const noexcept
    {
        return std::sqrt(length_squared());
    }

    vec3 vec3::normalize() const noexcept
    {
        const __m128 a       = _mm_load_ps(data);
        const __m128 d       = _mm_dp_ps(a, a, 0x7F);
        const __m128 inv_len = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(d));

        vec3 out;
        _mm_store_ps(out.data, _mm_mul_ps(a, inv_len));
        out.w = 1.0f;
        return out;
    }

    vec3 vec3::cross(const vec3& rhs) const noexcept
    {
        return {
            (y * rhs.z) - (z * rhs.y),
            (z * rhs.x) - (x * rhs.z),
            (x * rhs.y) - (y * rhs.x),
            1.0f
        };
    }

    float vec3::dot(const vec3& rhs) const noexcept
    {
        return (x * rhs.x) + (y * rhs.y) + (z * rhs.z);
    }

    matrix::matrix() noexcept { identity(); }

    matrix::matrix(const float m00, const float m01, const float m02, const float m03,
                   const float m10, const float m11, const float m12, const float m13,
                   const float m20, const float m21, const float m22, const float m23,
                   const float m30, const float m31, const float m32, const float m33) noexcept
    {
        a[0][0] = m00; a[0][1] = m01; a[0][2] = m02; a[0][3] = m03;
        a[1][0] = m10; a[1][1] = m11; a[1][2] = m12; a[1][3] = m13;
        a[2][0] = m20; a[2][1] = m21; a[2][2] = m22; a[2][3] = m23;
        a[3][0] = m30; a[3][1] = m31; a[3][2] = m32; a[3][3] = m33;
    }

    matrix::matrix(const matrix& rhs) noexcept
    {
        std::memcpy(m, rhs.m, sizeof(m));
    }

    matrix& matrix::operator=(const matrix& rhs) noexcept
    {
        std::memcpy(m, rhs.m, sizeof(m));
        return *this;
    }

    void matrix::identity() noexcept
    {
        std::memset(m, 0, sizeof(m));
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    matrix matrix::transpose() const noexcept
    {
        return {
            a[0][0], a[1][0], a[2][0], a[3][0],
            a[0][1], a[1][1], a[2][1], a[3][1],
            a[0][2], a[1][2], a[2][2], a[3][2],
            a[0][3], a[1][3], a[2][3], a[3][3]
        };
    }

    float& matrix::operator[](const int index) noexcept { return m[index]; }

    matrix matrix::mul(const matrix& rhs) const noexcept
    {
        matrix out;

        const __m128 b0 = _mm_load_ps(&rhs.m[0]);
        const __m128 b1 = _mm_load_ps(&rhs.m[4]);
        const __m128 b2 = _mm_load_ps(&rhs.m[8]);
        const __m128 b3 = _mm_load_ps(&rhs.m[12]);

        for (int i = 0; i < 4; ++i)
        {
            const __m128 a0 = _mm_set1_ps(m[i * 4 + 0]);
            const __m128 a1 = _mm_set1_ps(m[i * 4 + 1]);
            const __m128 a2 = _mm_set1_ps(m[i * 4 + 2]);
            const __m128 a3 = _mm_set1_ps(m[i * 4 + 3]);

            __m128 row = _mm_mul_ps(a0, b0);
            row = _mm_add_ps(row, _mm_mul_ps(a1, b1));
            row = _mm_add_ps(row, _mm_mul_ps(a2, b2));
            row = _mm_add_ps(row, _mm_mul_ps(a3, b3));

            _mm_store_ps(&out.m[i * 4], row);
        }
        return out;
    }

    matrix matrix::operator*(const matrix& rhs) const noexcept
    {
        return mul(rhs);
    }

    vec3 matrix::mul_vec(const vec3& v) const noexcept
    {
        return {
            (v.x * m[0] + v.y * m[1] + v.z * m[2]),
            (v.x * m[4] + v.y * m[5] + v.z * m[6]),
            (v.x * m[8] + v.y * m[9] + v.z * m[10]),
            1.0f
        };
    }

    vec3 matrix::mul_position(const vec3 &v) const noexcept
    {
        return {
            (v.x * m[0] + v.y * m[1] + v.z * m[2])  + m[3],
            (v.x * m[4] + v.y * m[5] + v.z * m[6])  + m[7],
            (v.x * m[8] + v.y * m[9] + v.z * m[10]) + m[11],
            1.0f
        };
    }

    vec3 matrix::mul_position_and_perspective_divide(const vec3 &v) const noexcept
    {
        const vec3 p(
        (v.x * m[0] + v.y * m[1] + v.z * m[2])  + m[3],
        (v.x * m[4] + v.y * m[5] + v.z * m[6])  + m[7],
        (v.x * m[8] + v.y * m[9] + v.z * m[10]) + m[11]);

        float h = (m[12] * v.x) + (m[13] * v.y) + (m[14] * v.z) + m[15];
        h = 1.0f / h;
        return (p * h);
    }

    matrix matrix::invert() const noexcept
    {
        matrix inv;
        inv[0]  =  m[5] * m[10] * m[15] - m[5] * m[11] * m[14] -
                   m[9] * m[6]  * m[15] + m[9] * m[7]  * m[14] +
                   m[13]* m[6]  * m[11] - m[13]* m[7]  * m[10];
        inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] +
                   m[8] * m[6]  * m[15] - m[8] * m[7]  * m[14] -
                   m[12]* m[6]  * m[11] + m[12]* m[7]  * m[10];
        inv[8]  =  m[4] * m[9]  * m[15] - m[4] * m[11] * m[13] -
                   m[8] * m[5]  * m[15] + m[8] * m[7]  * m[13] +
                   m[12]* m[5]  * m[11] - m[12]* m[7]  * m[9];
        inv[12] = -m[4] * m[9]  * m[14] + m[4] * m[10] * m[13] +
                   m[8] * m[5]  * m[14] - m[8] * m[6]  * m[13] -
                   m[12]* m[5]  * m[10] + m[12]* m[6]  * m[9];
        inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] +
                   m[9] * m[2]  * m[15] - m[9] * m[3]  * m[14] -
                   m[13]* m[2]  * m[11] + m[13]* m[3]  * m[10];
        inv[5]  =  m[0] * m[10] * m[15] - m[0] * m[11] * m[14] -
                   m[8] * m[2]  * m[15] + m[8] * m[3]  * m[14] +
                   m[12]* m[2]  * m[11] - m[12]* m[3]  * m[10];
        inv[9]  = -m[0] * m[9]  * m[15] + m[0] * m[11] * m[13] +
                   m[8] * m[1]  * m[15] - m[8] * m[3]  * m[13] -
                   m[12]* m[1]  * m[11] + m[12]* m[3]  * m[9];
        inv[13] =  m[0] * m[9]  * m[14] - m[0] * m[10] * m[13] -
                   m[8] * m[1]  * m[14] + m[8] * m[2]  * m[13] +
                   m[12]* m[1]  * m[10] - m[12]* m[2]  * m[9];
        inv[2]  =  m[1] * m[6]  * m[15] - m[1] * m[7]  * m[14] -
                   m[5] * m[2]  * m[15] + m[5] * m[3]  * m[14] +
                   m[13]* m[2]  * m[7]  - m[13]* m[3]  * m[6];
        inv[6]  = -m[0] * m[6]  * m[15] + m[0] * m[7]  * m[14] +
                   m[4] * m[2]  * m[15] - m[4] * m[3]  * m[14] -
                   m[12]* m[2]  * m[7]  + m[12]* m[3]  * m[6];
        inv[10] =  m[0] * m[5]  * m[15] - m[0] * m[7]  * m[13] -
                   m[4] * m[1]  * m[15] + m[4] * m[3]  * m[13] +
                   m[12]* m[1]  * m[7]  - m[12]* m[3]  * m[5];
        inv[14] = -m[0] * m[5]  * m[14] + m[0] * m[6]  * m[13] +
                   m[4] * m[1]  * m[14] - m[4] * m[2]  * m[13] -
                   m[12]* m[1]  * m[6]  + m[12]* m[2]  * m[5];
        inv[3]  = -m[1] * m[6]  * m[11] + m[1] * m[7]  * m[10] +
                   m[5] * m[2]  * m[11] - m[5] * m[3]  * m[10] -
                   m[9] * m[2]  * m[7]  + m[9] * m[3]  * m[6];
        inv[7]  =  m[0] * m[6]  * m[11] - m[0] * m[7]  * m[10] -
                   m[4] * m[2]  * m[11] + m[4] * m[3]  * m[10] +
                   m[8] * m[2]  * m[7]  - m[8] * m[3]  * m[6];
        inv[11] = -m[0] * m[5]  * m[11] + m[0] * m[7]  * m[9]  +
                   m[4] * m[1]  * m[11] - m[4] * m[3]  * m[9]  -
                   m[8] * m[1]  * m[7]  + m[8] * m[3]  * m[5];
        inv[15] =  m[0] * m[5]  * m[10] - m[0] * m[6]  * m[9]  -
                   m[4] * m[1]  * m[10] + m[4] * m[2]  * m[9]  +
                   m[8] * m[1]  * m[6]  - m[8] * m[2]  * m[5];

        const float det = m[0] * inv[0] + m[1] * inv[4] +
                          m[2] * inv[8] + m[3] * inv[12];

        if (det == 0.0f)
        {
            inv.identity();
            return inv;
        }

        const __m128 inv_det = _mm_set1_ps(1.0f / det);
        for (int i = 0; i < 16; i += 4)
        {
            const __m128 r = _mm_load_ps(&inv.m[i]);
            _mm_store_ps(&inv.m[i], _mm_mul_ps(r, inv_det));
        }
        return inv;
    }

    void frame::from_vector(const vec3& n) noexcept
    {
        w = n.normalize();
        if (std::fabs(w.x) > std::fabs(w.y))
        {
            const float l = 1.0f / std::sqrt(w.x * w.x + w.z * w.z);
            u = vec3(w.z * l, 0.0f, -w.x * l);
        }
        else
        {
            const float l = 1.0f / std::sqrt(w.y * w.y + w.z * w.z);
            u = vec3(0.0f, w.z * l, -w.y * l);
        }
        v = math::cross(w, u);
    }

    void frame::from_vector_tangent(const vec3& n, const vec3& t) noexcept
    {
        w = n.normalize();
        u = t.normalize();
        v = math::cross(w, u);
    }

    vec3 frame::to_local(const vec3& vec) const noexcept
    {
        return {
            math::dot(vec, u),
            math::dot(vec, v),
            math::dot(vec, w),
            1.0f
        };
    }

    vec3 frame::to_world(const vec3& vec) const noexcept
    {
        return (u * vec.x) + (v * vec.y) + (w * vec.z);
    }

    namespace math
    {
        color apply_gamma(const color& c, float gamma) noexcept
        {
            const float inv_gamma = 1.0f / (gamma > 0.0f ? gamma : 2.2f);
            return {
                std::pow(saturate(c.red),   inv_gamma),
                std::pow(saturate(c.green), inv_gamma),
                std::pow(saturate(c.blue),  inv_gamma)
            };
        }

        vec3 min(const vec3& a, const vec3& b) noexcept
        {
            vec3 out;
            const __m128 va = _mm_load_ps(a.data);
            const __m128 vb = _mm_load_ps(b.data);
            _mm_store_ps(out.data, _mm_min_ps(va, vb));
            out.w = 1.0f;
            return out;
        }

        vec3 max(const vec3& a, const vec3& b) noexcept
        {
            vec3 out;
            const __m128 va = _mm_load_ps(a.data);
            const __m128 vb = _mm_load_ps(b.data);
            _mm_store_ps(out.data, _mm_max_ps(va, vb));
            out.w = 1.0f;
            return out;
        }
    } // namespace math

    namespace transform
    {
        matrix translation(const vec3& v) noexcept
        {
            matrix mat;
            mat.a[0][3] = v.x;
            mat.a[1][3] = v.y;
            mat.a[2][3] = v.z;
            return mat;
        }

        matrix scaling(const vec3& v) noexcept
        {
            matrix mat;
            mat.m[0]  = v.x;
            mat.m[5]  = v.y;
            mat.m[10] = v.z;
            return mat;
        }

        matrix rotate_x(const float radians) noexcept
        {
            matrix mat;
            const float ct = std::cos(radians);
            const float st = std::sin(radians);
            mat.m[5]  =  ct;
            mat.m[6]  = -st;
            mat.m[9]  =  st;
            mat.m[10] =  ct;
            return mat;
        }

        matrix rotate_y(const float radians) noexcept
        {
            matrix mat;
            const float ct = std::cos(radians);
            const float st = std::sin(radians);
            mat.m[0]  =  ct;
            mat.m[2]  =  st;
            mat.m[8]  = -st;
            mat.m[10] =  ct;
            return mat;
        }

        matrix rotate_z(const float radians) noexcept
        {
            matrix mat;
            const float ct = std::cos(radians);
            const float st = std::sin(radians);
            mat.m[0] =  ct;
            mat.m[1] = -st;
            mat.m[4] =  st;
            mat.m[5] =  ct;
            return mat;
        }

        matrix look_at(const vec3& from, const vec3& to, const vec3& up) noexcept
        {
            matrix mat;
            const vec3 dir    = (from - to).normalize();
            const vec3 left   = math::cross(up, dir).normalize();
            const vec3 new_up = math::cross(dir, left);

            mat.a[0][0] = left.x;   mat.a[0][1] = left.y;   mat.a[0][2] = left.z;
            mat.a[1][0] = new_up.x; mat.a[1][1] = new_up.y; mat.a[1][2] = new_up.z;
            mat.a[2][0] = dir.x;    mat.a[2][1] = dir.y;    mat.a[2][2] = dir.z;

            mat.a[0][3] = -math::dot(from, left);
            mat.a[1][3] = -math::dot(from, new_up);
            mat.a[2][3] = -math::dot(from, dir);
            mat.a[3][3] = 1.0f;
            return mat;
        }

        matrix perspective(const float near_plane, const float far_plane,
                           const float aspect,     const float fov_deg) noexcept
        {
            matrix pers;
            std::memset(pers.m, 0, sizeof(pers.m));

            const float t = 1.0f / std::tan(fov_deg * 0.5f *
                                            math::pi<float> / 180.0f);

            pers.a[0][0] =  t / aspect;
            pers.a[1][1] =  t;
            pers.a[2][2] = -far_plane / (far_plane - near_plane);
            pers.a[2][3] = -(far_plane * near_plane) / (far_plane - near_plane);
            pers.a[3][2] = -1.0f;
            return pers;
        }
    } // namespace transform

    namespace sampling
    {
        vec3 spherical_to_world(const float theta, const float phi) noexcept
        {
            const float sin_theta = std::sin(theta);
            return {
                std::cos(phi) * sin_theta,
                std::sin(phi) * sin_theta,
                std::cos(theta),
                1.0f
            };
        }

        float spherical_theta(const vec3& wi) noexcept
        {
            return std::acos(math::saturate((wi.z + 1.0f) * 0.5f) * 2.0f - 1.0f);
        }

        float spherical_phi(const vec3& wi) noexcept
        {
            const float p = std::atan2(wi.y, wi.x);
            return (p < 0.0f) ? p + (2.0f * math::pi<float>) : p;
        }
    } // namespace sampling

} // namespace fox_tracer
