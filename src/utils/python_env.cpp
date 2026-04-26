// COPIED FROM A PREVIOUS NDA PROJECT BUT I MADE SURE TO
// RESTRUCTURE AND CHANGE
// ANYTHING THE NDA DOCUMENT MENTIONED!
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
#include "utils/python_env.h"
#include "utils/helper.h"
#include "utils/logger.h"
#include "utils/paths.h"


#include <windows.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <thread>

// TODO: Add API Support
namespace fox_tracer::utility
{
    std::string python_env::resolve_venv_python() const
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path base = fs::path(paths::resolve("assets/scripts"));
        const fs::path candidates[] = {
            base / ".venv" / "Scripts" / "python.exe",
            base / ".venv" / "bin" / "python.exe",
            base / ".venv" / "bin" / "python",
        };
        for (const auto& p : candidates)
            if (fs::exists(p, ec)) return p.string();
        return {};
    }

    void python_env::create_venv()
    {
        namespace fs = std::filesystem;

        const std::string python = helper::find_venv_base_python();

        const fs::path venv_dir =
            fs::path(paths::resolve("assets/scripts")) / ".venv";

        std::error_code ec;
        if (resolve_venv_python().empty() && fs::exists(venv_dir, ec))
        {
            LOG_INFO("py_effect")
                << "create_venv: removing stale " << venv_dir.string();
            fs::remove_all(venv_dir, ec);
            if (ec)
            {
                LOG_WARN("py_effect")
                    << "create_venv: could not remove stale venv ("
                    << ec.message()
                    << ") can fail";
            }
        }

        std::string lower = python;
        std::ranges::transform(lower, lower.begin(),
           [](unsigned char c)
           {
               return (char)std::tolower(c);
           });

        if (lower.find("anaconda") != std::string::npos ||
            lower.find("miniconda") != std::string::npos ||
            lower.find("\\conda") != std::string::npos)
        {
            LOG_WARN("py_effect")
                << "create_venv: base interpreter looks like conda ("
                << python
                << ")";
        }

        std::string cmd;
        const bool is_py_launcher =
            python.rfind("py ", 0) == 0 || python == "py";
        if (is_py_launcher) cmd += python;
        else { cmd += '"'; cmd += python; cmd += '"'; }
        cmd += " -m venv --copies \"";
        cmd += venv_dir.string(); cmd += "\"";

        run_async(cmd, "create_venv", true);
    }

    void python_env::install_ai_deps_now(const std::string& index_url,
                                         const std::string& worker_python)
    {
        const std::string venv_py = resolve_venv_python();
        std::string python = venv_py;
        if (python.empty()) python = worker_python;
        if (python.empty()) python = "python";

        if (venv_py.empty())
        {
            LOG_WARN("py_effect")
                << "install_ai_deps: no .venv found installing into "
                << python
                << " create a local .venv first to avoid touching "
                   "the system interpreter";
        }
        else
        {
            LOG_INFO("py_effect")
                << "ai depns: target venv python = " << python
                << (index_url.empty()
                        ? " PyPI default wheel"
                        : std::string(" (index: ") + index_url + ")");
        }

        std::string torch_cmd;
        torch_cmd += '"'; torch_cmd += python;
        torch_cmd += "\" -m pip install --upgrade torch";
        if (!index_url.empty())
        {
            torch_cmd += " --index-url ";
            torch_cmd += index_url;
        }

        std::string rest_cmd;
        rest_cmd += '"'; rest_cmd += python;
        rest_cmd += "\" -m pip install --upgrade ";
        rest_cmd += "Pillow huggingface_hub numpy tqdm";

        std::string cmd = "cmd /c \"" + torch_cmd + " && " + rest_cmd + "\"";

        const bool worker_on_venv =
            !venv_py.empty() && worker_python == venv_py;
        run_async(cmd, "install_ai_deps",
                  !worker_on_venv);
    }

    bool python_env::cuda_available() const
    {
        if (!cuda_detection_done_) refresh_cuda_detection();
        return cuda_detected_cached_;
    }

    void python_env::refresh_cuda_detection() const
    {
        cuda_detection_done_ = true;

        char buf[1024];
        DWORD n = GetEnvironmentVariableA("CUDA_PATH", buf, sizeof(buf));
        if (n > 0 && n < sizeof(buf))
        {
            cuda_detected_cached_ = true;
            return;
        }

        char path_buf[32 * 1024];
        DWORD pn = GetEnvironmentVariableA("PATH", path_buf, sizeof(path_buf));
        if (pn == 0 || pn >= sizeof(path_buf))
        {
            cuda_detected_cached_ = false;
            return;
        }

        namespace fs = std::filesystem;
        std::error_code ec;
        std::string path(path_buf, pn);
        std::size_t start = 0;
        while (start <= path.size())
        {
            std::size_t end = path.find(';', start);
            if (end == std::string::npos) end = path.size();
            if (end > start)
            {
                fs::path p = fs::path(path.substr(start, end - start))
                           / "nvidia-smi.exe";
                if (fs::exists(p, ec))
                {
                    cuda_detected_cached_ = true;
                    return;
                }
            }
            start = end + 1;
        }
        cuda_detected_cached_ = false;
    }

    void python_env::run_async(std::string command_line,
                               std::string log_tag,
                               bool restart_worker_on_success)
    {
        if (busy_.exchange(true)) return;
        {
            std::lock_guard<std::mutex> lk(last_summary_mtx_);
            last_summary_ = log_tag + ": running...";
        }
        LOG_INFO("py_effect") << log_tag << ": " << command_line;

        std::thread([this,
                     cmd  = std::move(command_line),
                     tag  = std::move(log_tag),
                     auto_restart = restart_worker_on_success]() mutable
        {
            STARTUPINFOA si{};
            si.cb = sizeof(si);
            PROCESS_INFORMATION pi{};
            BOOL ok = CreateProcessA(
                nullptr, cmd.data(),
                nullptr, nullptr, FALSE,
                CREATE_NEW_CONSOLE,
                nullptr, nullptr,
                &si, &pi);
            std::string summary;
            bool success = false;
            if (!ok)
            {
                summary = tag + ": CreateProcess failed ("
                        + std::to_string(GetLastError()) + ")";
                LOG_ERROR("py_effect") << summary;
            }
            else
            {
                WaitForSingleObject(pi.hProcess, INFINITE);
                DWORD code = 1;
                GetExitCodeProcess(pi.hProcess, &code);
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
                success = (code == 0);
                summary = tag + (success ? ": success"
                                         : ": exit code "
                                           + std::to_string(code));
                LOG_INFO("py_effect") << summary;
            }
            {
                std::lock_guard<std::mutex> lk(last_summary_mtx_);
                last_summary_ = std::move(summary);
            }
            if (success && auto_restart) pending_worker_restart_.store(true);
            busy_.store(false);
        }).detach();
    }
} // namespace fox_tracer::utility

