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
#include "utils/helper.h"


#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace fox_tracer::helper
{
    std::string json_escape(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (const char c : s)
        {
            if (c == '"' || c == '\\' || c == '\r') continue;
            if (c == '\n')
            {
                out.push_back(' '); continue;
            }
            out.push_back(c);
        }
        return out;
    }

    std::string vec3_str(float x, float y, float z)
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%g %g %g", x, y, z);
        return buf;
    }

    std::string float_str(float v)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%g", v);
        return buf;
    }

    std::string fmt_label(const std::string& prefix, int counter)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s-%03d", prefix.c_str(), counter);
        return buf;
    }

    bool ends_with_ci(const std::string& s, const std::string& suffix)
    {
        if (s.size() < suffix.size()) return false;
        const std::size_t off = s.size() - suffix.size();

        for (std::size_t i = 0; i < suffix.size(); ++i)
        {
            const char a = static_cast<char>(std::tolower(
                static_cast<unsigned char>(s[off + i])));

            const char b = static_cast<char>(std::tolower(
                static_cast<unsigned char>(suffix[i])));

            if (a != b) return false;
        }
        return true;
    }

    std::string leaf_name(const std::string& p)
    {
        const std::size_t slash = p.find_last_of("/\\");
        return (slash == std::string::npos) ? p : p.substr(slash + 1);
    }

    int clamp_tile_size(const int v)
    {
        return std::clamp(v, 8, 512);
    }

    float power_heuristic(const float pdf_a, const  float pdf_b) noexcept
    {
        if (pdf_a <= 0.0f) return 0.0f;

        const float a = pdf_a * pdf_a;
        const float b = pdf_b * pdf_b;

        const float s = a + b;
        if (s <= 0.0f || !std::isfinite(s)) return 0.0f;

        return a / s;
    }

    bool has_scene_json(const std::string& dir)
    {
        const std::string p = dir + "/scene.json";
        const DWORD a = GetFileAttributesA(p.c_str());

        return a != INVALID_FILE_ATTRIBUTES
            && !(a & FILE_ATTRIBUTE_DIRECTORY);
    }

    void list_subdirs(const std::string& dir, std::vector<std::string>& out)
    {
        const std::string pattern = dir + "/*";
        WIN32_FIND_DATAA fd;

        const HANDLE h = FindFirstFileA(pattern.c_str(), &fd);

        if (h == INVALID_HANDLE_VALUE) return;
        do
        {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
            const char* n = fd.cFileName;
            if (n[0] == '.' && (n[1] == 0 || (n[1] == '.' && n[2] == 0))) continue;
            out.push_back(dir + "/" + n);
        } while (FindNextFileA(h, &fd));

        FindClose(h);
    }

    void list_files(const std::string& dir,
                    const std::vector<std::string>& extensions,
                    std::vector<std::string>& out)
    {
        const std::string pattern = dir + "/*";
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) return;
        do
        {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            const std::string name = fd.cFileName;
            bool ok = extensions.empty();
            for (const auto& ext : extensions)
            {
                if (name.size() >= ext.size()
                    && _stricmp(name.c_str() + name.size() - ext.size(),
                                ext.c_str()) == 0)
                {
                    ok = true; break;
                }
            }
            if (ok) out.push_back(name);
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }

    std::string unescape_json(const std::string_view in)
    {
        std::string out;
        out.reserve(in.size());
        for (std::size_t i = 0; i < in.size(); ++i)
        {
            const char c = in[i];
            if (c != '\\' || i + 1 >= in.size())
            {
                out.push_back(c);
                continue;
            }
            char n = in[++i];
            switch (n)
            {
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            case 'n':  out.push_back('\n'); break;
            case 't':  out.push_back('\t'); break;
            case 'r':  out.push_back('\r'); break;
            case 'b':  out.push_back('\b'); break;
            case 'f':  out.push_back('\f'); break;
            default:   out.push_back(n); break;
            }
        }
        return out;
    }

    std::size_t json_find_key(const std::string_view src, const std::string_view key)
    {
        std::string pat;
        pat.reserve(key.size() + 4);
        pat += '"';
        pat.append(key.data(), key.size());
        pat += '"';
        std::size_t pos = 0;
        while (true)
        {
            pos = src.find(pat, pos);
            if (pos == std::string_view::npos) return std::string_view::npos;
            std::size_t back = pos;
            while (back > 0 && std::isspace(static_cast<unsigned char>(src[back - 1])))
                --back;
            if (back == 0 || src[back - 1] == '{' || src[back - 1] == ',')
            {
                std::size_t after = pos + pat.size();
                while (after < src.size() &&
                       std::isspace(static_cast<unsigned char>(src[after])))
                    ++after;
                if (after < src.size() && src[after] == ':') return after + 1;
            }
            pos += pat.size();
        }
    }

    bool json_extract_string(const std::string_view src, const std::string_view key,
                             std::string& out)
    {
        std::size_t p = json_find_key(src, key);
        if (p == std::string_view::npos) return false;
        while (p < src.size() &&
               std::isspace(static_cast<unsigned char>(src[p])))
            ++p;
        if (p >= src.size() || src[p] != '"') return false;
        ++p;
        const std::size_t start = p;
        while (p < src.size())
        {
            if (src[p] == '\\' && p + 1 < src.size()) { p += 2; continue; }
            if (src[p] == '"') break;
            ++p;
        }
        if (p >= src.size()) return false;
        out = unescape_json(src.substr(start, p - start));
        return true;
    }

    bool json_extract_int(const std::string_view src, const std::string_view key,
                          long long& out)
    {
        std::size_t p = json_find_key(src, key);
        if (p == std::string_view::npos) return false;
        while (p < src.size() &&
               std::isspace(static_cast<unsigned char>(src[p])))
            ++p;
        const std::size_t start = p;
        if (p < src.size() && (src[p] == '-' || src[p] == '+')) ++p;
        while (p < src.size() &&
               std::isdigit(static_cast<unsigned char>(src[p])))
            ++p;
        if (p == start) return false;
        try
        {
            out = std::stoll(std::string(src.substr(start, p - start)));
        }
        catch (...) { return false; }
        return true;
    }

    bool json_extract_double(
        const std::string_view src,
        const std::string_view key,
        double& out)
    {
        std::size_t p = json_find_key(src, key);
        if (p == std::string_view::npos) return false;
        while (p < src.size() &&
               std::isspace(static_cast<unsigned char>(src[p])))
            ++p;
        const std::size_t start = p;
        if (p < src.size() && (src[p] == '-' || src[p] == '+')) ++p;
        while (p < src.size() &&
               (std::isdigit(static_cast<unsigned char>(src[p])) ||
                src[p] == '.' || src[p] == 'e' || src[p] == 'E' ||
                src[p] == '+' || src[p] == '-'))
            ++p;
        if (p == start) return false;
        try
        {
            out = std::stod(std::string(src.substr(start, p - start)));
        }
        catch (...) { return false; }
        return true;
    }

    bool json_extract_string_array(const std::string_view src,const  std::string_view key,
                                   std::vector<std::string>& out)
    {
        std::size_t p = json_find_key(src, key);
        if (p == std::string_view::npos) return false;
        while (p < src.size() &&
               std::isspace(static_cast<unsigned char>(src[p])))
            ++p;
        if (p >= src.size() || src[p] != '[') return false;
        ++p;
        out.clear();
        while (p < src.size())
        {
            while (p < src.size() &&
                   (std::isspace(static_cast<unsigned char>(src[p])) ||
                    src[p] == ','))
                ++p;
            if (p >= src.size()) return false;
            if (src[p] == ']') return true;
            if (src[p] != '"') return false;
            ++p;
            std::size_t start = p;
            while (p < src.size())
            {
                if (src[p] == '\\' && p + 1 < src.size()) { p += 2; continue; }
                if (src[p] == '"') break;
                ++p;
            }
            if (p >= src.size()) return false;
            out.push_back(unescape_json(src.substr(start, p - start)));
            ++p;
        }
        return false;
    }

    bool pipe_read_exact(void* handle, std::uint8_t* p, std::size_t n)
    {
        auto h = static_cast<HANDLE>(handle);
        while (n > 0)
        {
            DWORD got = 0;
            BOOL ok = ReadFile(h, p, static_cast<DWORD>(n), &got, nullptr);
            if (!ok || got == 0) return false;
            p += got;
            n -= got;
        }
        return true;
    }

    bool pipe_read_line(void* handle, std::string& out)
    {
        auto h = static_cast<HANDLE>(handle);
        out.clear();
        char c;
        DWORD got = 0;
        while (true)
        {
            BOOL ok = ReadFile(h, &c, 1, &got, nullptr);
            if (!ok || got == 0) return false;
            if (c == '\n') return true;
            if (c == '\r') continue;
            out.push_back(c);
        }
    }

    std::string pick_python(const std::string& script_path)
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path scripts_dir = fs::path(script_path).parent_path();

        const fs::path win_venv = scripts_dir / ".venv" / "Scripts" / "python.exe";
        if (fs::exists(win_venv, ec)) return win_venv.string();

        const fs::path posix_venv_exe =
            scripts_dir / ".venv" / "bin" / "python.exe";
        if (fs::exists(posix_venv_exe, ec)) return posix_venv_exe.string();
        const fs::path posix_venv =
            scripts_dir / ".venv" / "bin" / "python";
        if (fs::exists(posix_venv, ec)) return posix_venv.string();

        char buf[MAX_PATH];
        DWORD n = GetEnvironmentVariableA("FOX_TRACER_PYTHON", buf, MAX_PATH);
        if (n > 0 && n < MAX_PATH) return std::string(buf);

        return "python";
    }

    std::string build_python_command_line(const std::string& python_exe,
                                          const std::string& script_path)
    {
        std::string cmd;
        cmd.reserve(python_exe.size() + script_path.size() + 16);
        cmd += '"'; cmd += python_exe; cmd += '"';
        cmd += " -u ";
        cmd += '"'; cmd += script_path; cmd += '"';
        return cmd;
    }

    std::string find_venv_base_python()
    {
        char buf[1024];
        DWORD n = GetEnvironmentVariableA("FOX_TRACER_PYTHON", buf, 1024);
        if (n > 0 && n < 1024) return std::string(buf);

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;
        PROCESS_INFORMATION pi{};
        char probe[] = "py -3 -c \"import sys\"";
        if (CreateProcessA(nullptr, probe, nullptr, nullptr, FALSE,
                           CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
        {
            WaitForSingleObject(pi.hProcess, 5000);
            DWORD code = 1;
            GetExitCodeProcess(pi.hProcess, &code);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            if (code == 0) return "py -3";
        }

        return "python";
    }
} // namespace fox_tracer::helper
