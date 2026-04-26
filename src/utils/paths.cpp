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
#include "utils/paths.h"

#include <windows.h>
#include <filesystem>

namespace fox_tracer::paths
{
    namespace fs = std::filesystem;

    namespace
    {
        std::string compute_exe_dir()
        {
            std::wstring buf(MAX_PATH, L'\0');
            for (;;)
            {
                const DWORD n = GetModuleFileNameW(nullptr, buf.data(),
                                                   static_cast<DWORD>(buf.size()));
                if (n == 0) return {};
                if (n < buf.size())
                {
                    buf.resize(n);
                    break;
                }
                buf.resize(buf.size() * 2);
            }

            try
            {
                return fs::path(buf).parent_path().string();
            }
            catch (...)
            {
                return {};
            }
        }
    } // namespace

    const std::string& exe_dir()
    {
        static const std::string cached = compute_exe_dir();
        return cached;
    }

    bool exists(const std::string& path)
    {
        std::error_code ec;
        return fs::exists(fs::path(path), ec);
    }

    namespace
    {
        std::string normalize(const fs::path& p)
        {
            return p.lexically_normal().make_preferred().string();
        }
    }

    std::string resolve(const std::string& input)
    {
        if (input.empty()) return input;

        std::error_code ec;
        const fs::path p(input);
        if (p.is_absolute())
        {
            return normalize(p);
        }

        const std::string& exe = exe_dir();
        if (!exe.empty())
        {
            const fs::path exe_path(exe);
            const auto candidate = exe_path / p;
            if (fs::exists(candidate, ec))
            {
                return normalize(candidate);
            }

            const auto parent_candidate = exe_path.parent_path() / p;
            if (fs::exists(parent_candidate, ec))
            {
                return normalize(parent_candidate);
            }
        }

        if (fs::exists(p, ec))
        {
            return normalize(p);
        }

        if (!exe.empty())
        {
            return normalize(fs::path(exe) / p);
        }
        return normalize(p);
    }
} // namespace fox_tracer::paths
