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
#include "utils/logger.h"

#include <windows.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace fox_tracer::log
{
    namespace
    {
        std::atomic<level> g_min_level{level::info};
        std::mutex         g_write_mutex;
        std::atomic<bool>  g_colour_enabled{false};

        bool enable_vt_on(HANDLE h)
        {
            if (h == INVALID_HANDLE_VALUE || h == nullptr) return false;
            DWORD mode = 0;
            if (!GetConsoleMode(h, &mode)) return false;
            if ((mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0) return true;
            return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
        }

        const char* colour_for(level l) noexcept
        {
            switch (l)
            {
                case level::debug: return "\x1b[90m"; //~ bright black / grey
                case level::info:  return "\x1b[36m"; //~ cyan
                case level::warn:  return "\x1b[33m"; //~ yellow
                case level::error: return "\x1b[1;31m"; //~ bold red
            }
            return "";
        }

        const char* label_for(level l) noexcept
        {
            switch (l)
            {
                case level::debug: return "DEBUG";
                case level::info:  return "INFO ";
                case level::warn:  return "WARN ";
                case level::error: return "ERROR";
            }
            return "?????";
        }

        std::string timestamp()
        {
            using namespace std::chrono;
            const auto now = system_clock::now();
            const auto ms  = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
            const std::time_t t = system_clock::to_time_t(now);
            std::tm tm{};
            localtime_s(&tm, &t);

            std::ostringstream os;
            os << std::setfill('0')
               << std::setw(2) << tm.tm_hour << ':'
               << std::setw(2) << tm.tm_min  << ':'
               << std::setw(2) << tm.tm_sec  << '.'
               << std::setw(3) << ms.count();
            return os.str();
        }
    } // namespace

    void init()
    {
        SetConsoleOutputCP(CP_UTF8);

        const bool out_ok = enable_vt_on(GetStdHandle(STD_OUTPUT_HANDLE));
        const bool err_ok = enable_vt_on(GetStdHandle(STD_ERROR_HANDLE));
        g_colour_enabled.store(out_ok && err_ok, std::memory_order_relaxed);
    }

    void set_min_level(level l) noexcept
    {
        g_min_level.store(l, std::memory_order_relaxed);
    }

    void write(level l, std::string_view tag, std::string_view msg)
    {
        if (static_cast<int>(l) < static_cast<int>(g_min_level.load(std::memory_order_relaxed)))
        {
            return;
        }

        const std::string ts     = timestamp();
        const bool        colour = g_colour_enabled.load(std::memory_order_relaxed);

        FILE* sink = (l == level::error || l == level::warn) ? stderr : stdout;

        std::ostringstream line;
        if (colour)
        {
            line << "\x1b[90m" << ts << "\x1b[0m "
                 << colour_for(l) << '[' << label_for(l) << "]\x1b[0m "
                 << "\x1b[90m" << tag << "\x1b[0m "
                 << msg;
        }
        else
        {
            line << ts << " [" << label_for(l) << "] " << tag << ' ' << msg;
        }
        const std::string out = line.str();

        std::lock_guard<std::mutex> lock(g_write_mutex);
        std::fwrite(out.data(), 1, out.size(), sink);
        std::fputc('\n', sink);
        std::fflush(sink);
    }
} // namespace fox_tracer::log
