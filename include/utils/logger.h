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
#ifndef RAYTRACER_WITH_AI_LOGGER_H
#define RAYTRACER_WITH_AI_LOGGER_H

#include <sstream>
#include <string>
#include <string_view>

namespace fox_tracer::log
{
    enum class level
    {
        debug,
        info,
        warn,
        error
    };

    void init();
    void set_min_level(level l) noexcept;
    void write(level l, std::string_view tag, std::string_view msg);

    namespace detail
    {
        class line
        {
        public:
            line(const level l, const std::string_view tag) noexcept
            : level_(l), tag_(tag) {}

            ~line() { write(level_, tag_, stream_.str()); }

            line(const line&)            = delete;
            line& operator=(const line&) = delete;

            template <typename T>
            line& operator<<(const T& value)
            {
                stream_ << value;
                return *this;
            }
        private:
            level             level_;
            std::string_view  tag_;
            std::ostringstream stream_;
        };
    }
} // namespace fox_tracer::log

#define FOX_LOG(level, tag) ::fox_tracer::log::detail::line(level, tag)
#define LOG_DEBUG(tag) FOX_LOG(::fox_tracer::log::level::debug, tag)
#define LOG_INFO(tag)  FOX_LOG(::fox_tracer::log::level::info,  tag)
#define LOG_WARN(tag)  FOX_LOG(::fox_tracer::log::level::warn,  tag)
#define LOG_ERROR(tag) FOX_LOG(::fox_tracer::log::level::error, tag)

#endif //RAYTRACER_WITH_AI_LOGGER_H
