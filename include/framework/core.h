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
#ifndef RAYTRACER_WITH_AI_CORE_H
#define RAYTRACER_WITH_AI_CORE_H

#include <type_traits>

namespace fox_tracer
{
    namespace math
    {
        template<typename T>
        inline constexpr T pi = T(3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679);

        template<typename T>
        inline constexpr T inverse_pi = T(0.31830988618379067153776752674502872406891929148054689697427663658793739166691358753951457096296997632);

        template<typename T>
        inline constexpr T two_pi = T(2.0 * pi<T>);

        template<typename T>
        inline constexpr T two_pi_inverse = T(1.0 / two_pi<T>);

        template<typename T>
        inline constexpr T four_pi_inverse = T(1.0 / (4.0 * pi<T>));

        template<typename T>
        inline constexpr T epsilon = T(0.000001);

        template<typename T>
        requires std::is_arithmetic_v<T>
        constexpr T squared(T x) { return x * x; }

        [[nodiscard]] constexpr float saturate(float v) noexcept
        {
            return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
    }

    class color
    {
    public:
        union
        {
            struct { float _r, _g, _b; };
            struct { float red, green, blue; };
            float data[3];
        };

         color() noexcept;
        ~color() noexcept = default;

        color(float r, float g, float b)                         noexcept;
        color(unsigned char r, unsigned char g, unsigned char b) noexcept;

        color(const color&) noexcept = default;
        color(color&&)      noexcept = default;

        color& operator=(const color&) noexcept = default;
        color& operator=(color&&)      noexcept = default;

        void to_rgb(unsigned char& r, unsigned char& g, unsigned char& b) const noexcept;

        //~ overloading
        color operator+(const color& rhs) const noexcept;
        color operator-(const color& rhs) const noexcept;
        color operator*(const color& rhs) const noexcept;
        color operator/(const color& rhs) const noexcept;
        color operator*(float v)          const noexcept;
        color operator/(float v)          const noexcept;

        [[nodiscard]] float luminance() const noexcept;
    };

    class alignas(16) vec3
    {
    public:
        union
        {
            struct { float x, y, z, w; };
            float data[4];
        };

         vec3() noexcept;
        ~vec3() noexcept = default;

        vec3(float x, float y, float z, float w = 0.0f) noexcept;

        //~ vector to vector overloading
        vec3 operator+(const vec3& rhs) const noexcept;
        vec3 operator-(const vec3& rhs) const noexcept;
        vec3 operator*(const vec3& rhs) const noexcept;

        //~ scalar and vector
        vec3 operator*(float scalar) const noexcept;
        vec3 operator/(float scalar) const noexcept;
        vec3 operator-() const noexcept;

        //~ helpers
        [[nodiscard]] vec3  perspective_divide() const noexcept;
        [[nodiscard]] float length_squared    () const noexcept;
        [[nodiscard]] float length            () const noexcept;
        [[nodiscard]] vec3  normalize         () const noexcept;

        [[nodiscard]] vec3  cross(const vec3& rhs) const noexcept;
        [[nodiscard]] float dot  (const vec3& rhs) const noexcept;
    };

    // TODO: Benchmark it if needed make it cacheline fit
    struct vertex
    {
        vec3 point {};
        vec3 normal{};

        float u{};
        float v{};
    };

    // Row major matrix
    class alignas(16) matrix
    {
    public:
        union
        {
            float a[4][4];
            float m[16];
        };

        matrix() noexcept;
        matrix(float m00, float m01, float m02, float m03,
               float m10, float m11, float m12, float m13,
               float m20, float m21, float m22, float m23,
               float m30, float m31, float m32, float m33) noexcept;

                      void   identity ()       noexcept;
        [[nodiscard]] matrix transpose() const noexcept;
        [[nodiscard]] matrix invert   () const noexcept;

        float& operator[](int index) noexcept;

        [[nodiscard]]
        matrix mul(const matrix& rhs)       const noexcept;
        matrix operator*(const matrix& rhs) const noexcept;

        matrix(const matrix& rhs)            noexcept;
        matrix& operator=(const matrix& rhs) noexcept;

        [[nodiscard]] vec3 mul_vec  (const vec3& v) const noexcept;
        [[nodiscard]] vec3 mul_point(const vec3& v) const noexcept;

        [[nodiscard]] vec3 mul_point_and_perspective_divide(const vec3& v) const noexcept;
    };

    class frame
    {
    public:
        vec3 u, v, w;

        void from_vector        (const vec3& n) noexcept;
        void from_vector_tangent(const vec3& n,
                                 const vec3& t) noexcept;

        [[nodiscard]] vec3 to_local(const vec3& vec) const noexcept;
        [[nodiscard]] vec3 to_world(const vec3& vec) const noexcept;
    };

    namespace math
    {
        [[nodiscard]] color apply_gamma(const color& c, float gamma) noexcept;

        [[nodiscard]] __forceinline
        float dot(const vec3& a, const vec3& b)  noexcept { return a.dot(b); }

        [[nodiscard]] __forceinline
        vec3 cross(const vec3& a, const vec3& b) noexcept { return a.cross(b); }

        [[nodiscard]] vec3 min(const vec3& a, const vec3& b) noexcept;
        [[nodiscard]] vec3 max(const vec3& a, const vec3& b) noexcept;
    }

    namespace transform
    {
        [[nodiscard]] matrix translation(const vec3& v) noexcept;
        [[nodiscard]] matrix scaling    (const vec3& v) noexcept;
        [[nodiscard]] matrix rotate_x   (float radians) noexcept;
        [[nodiscard]] matrix rotate_y   (float radians) noexcept;
        [[nodiscard]] matrix rotate_z   (float radians) noexcept;

        [[nodiscard]] matrix look_at(const vec3& from,
                                     const vec3& to,
                                     const vec3& up) noexcept;

        [[nodiscard]] matrix perspective(float near_plane, float far_plane,
                                         float aspect,     float fov_deg) noexcept;
    }

    namespace sampling
    {
        [[nodiscard]] vec3  spherical_to_world(float theta, float phi) noexcept;
        [[nodiscard]] float spherical_theta   (const vec3& wi) noexcept;
        [[nodiscard]] float spherical_phi     (const vec3& wi) noexcept;
    }

}

#endif //RAYTRACER_WITH_AI_CORE_H
