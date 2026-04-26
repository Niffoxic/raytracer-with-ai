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

#include "utils/effect_runner.h"
#include "utils/helper.h"
#include "utils/logger.h"


#include <windows.h>
#include <algorithm>
#include <cstring>
#include <string>

namespace fox_tracer::utility
{

    effect_runner::effect_runner() noexcept = default;

    effect_runner::~effect_runner()
    {
        shutdown();
    }

    bool effect_runner::start(const std::string& script_path)
    {
        if (alive_.load()) return true;

        {
            std::lock_guard<std::mutex> lk(status_mtx_);
            script_path_ = script_path;
        }

        SECURITY_ATTRIBUTES sa{};
        sa.nLength        = sizeof(sa);
        sa.lpSecurityDescriptor = nullptr;
        sa.bInheritHandle = TRUE;

        HANDLE in_r = nullptr, in_w = nullptr;
        HANDLE out_r = nullptr, out_w = nullptr;
        HANDLE err_r = nullptr, err_w = nullptr;
        if (!CreatePipe(&in_r, &in_w, &sa, 0))
        {
            std::lock_guard<std::mutex> lk(status_mtx_);
            last_error_ = "CreatePipe(stdin) failed";
            return false;
        }
        if (!CreatePipe(&out_r, &out_w, &sa, 0))
        {
            CloseHandle(in_r); CloseHandle(in_w);
            std::lock_guard<std::mutex> lk(status_mtx_);
            last_error_ = "CreatePipe(stdout) failed";
            return false;
        }
        if (!CreatePipe(&err_r, &err_w, &sa, 0))
        {
            CloseHandle(in_r);  CloseHandle(in_w);
            CloseHandle(out_r); CloseHandle(out_w);
            std::lock_guard<std::mutex> lk(status_mtx_);
            last_error_ = "CreatePipe(stderr) failed";
            return false;
        }

        SetHandleInformation(in_w,  HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si{};
        si.cb         = sizeof(si);
        si.dwFlags    = STARTF_USESTDHANDLES;
        si.hStdInput  = in_r;
        si.hStdOutput = out_w;
        si.hStdError  = err_w;

        const std::string python_exe = helper::pick_python(script_path);
        {
            std::lock_guard<std::mutex> lk(status_mtx_);
            picked_python_ = python_exe;
        }
        std::string cmd = helper::build_python_command_line(python_exe, script_path);
        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessA(
            nullptr,
            cmd.data(),
            nullptr, nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            nullptr, nullptr,
            &si, &pi);

        CloseHandle(in_r);
        CloseHandle(out_w);
        CloseHandle(err_w);

        if (!ok)
        {
            DWORD err = GetLastError();
            CloseHandle(in_w);
            CloseHandle(out_r);
            CloseHandle(err_r);
            std::lock_guard<std::mutex> lk(status_mtx_);
            last_error_ = "CreateProcess failed, error " + std::to_string(err)
                        + " (cmd: " + cmd + ")";
            LOG_ERROR("py_effect") << last_error_;
            return false;
        }
        CloseHandle(pi.hThread);

        h_process_  = pi.hProcess;
        h_stdin_w_  = in_w;
        h_stdout_r_ = out_r;
        h_stderr_r_ = err_r;

        alive_.store(true);
        stopping_.store(false);
        status_.store(status::idle);

        reader_        = std::thread([this]{ reader_loop(); });
        stderr_reader_ = std::thread([this]{ stderr_loop(); });

        LOG_INFO("py_effect") << "spawned worker: " << cmd;
        return true;
    }

    void effect_runner::shutdown()
    {
        if (!alive_.load() && !reader_.joinable()) return;

        stopping_.store(true);

        if (h_stdin_w_ != nullptr)
        {
            write_line("{\"op\":\"shutdown\"}");
            CloseHandle(h_stdin_w_);
            h_stdin_w_ = nullptr;
        }

        if (h_process_ != nullptr)
        {
            DWORD wr = WaitForSingleObject(h_process_, 1500);
            if (wr != WAIT_OBJECT_0) force_kill();
            CloseHandle(h_process_);
            h_process_ = nullptr;
        }

        if (h_stdout_r_ != nullptr)
        {
            CloseHandle(h_stdout_r_);
            h_stdout_r_ = nullptr;
        }
        if (h_stderr_r_ != nullptr)
        {
            CloseHandle(h_stderr_r_);
            h_stderr_r_ = nullptr;
        }

        alive_.store(false);
        if (reader_.joinable())        reader_.join();
        if (stderr_reader_.joinable()) stderr_reader_.join();
    }

    void effect_runner::force_kill()
    {
        if (h_process_ == nullptr) return;
        TerminateProcess(h_process_, 1);
        WaitForSingleObject(h_process_, 500);
    }

    bool effect_runner::write_line(const std::string& line)
    {
        std::lock_guard<std::mutex> lk(write_mtx_);
        if (h_stdin_w_ == nullptr) return false;
        std::string out = line;
        if (out.empty() || out.back() != '\n') out.push_back('\n');
        DWORD written = 0;
        BOOL ok = WriteFile(h_stdin_w_, out.data(),
                            static_cast<DWORD>(out.size()),
                            &written, nullptr);
        if (!ok || written != out.size())
        {
            std::lock_guard<std::mutex> lk2(status_mtx_);
            last_error_ = "WriteFile(stdin) failed";
            return false;
        }
        return true;
    }

    bool effect_runner::write_bytes(const std::uint8_t* p, std::size_t n)
    {
        std::lock_guard<std::mutex> lk(write_mtx_);
        if (h_stdin_w_ == nullptr) return false;

        while (n > 0)
        {
            DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(n, 1 << 20));
            DWORD written = 0;
            if (!WriteFile(h_stdin_w_, p, chunk, &written, nullptr) ||
                written == 0)
            {
                std::lock_guard<std::mutex> lk2(status_mtx_);
                last_error_ = "WriteFile(payload) failed";
                return false;
            }
            p += written;
            n -= written;
        }
        return true;
    }

    bool effect_runner::request_list()
    {
        return write_line("{\"op\":\"list\"}");
    }

    bool effect_runner::request_load(const std::string& name)
    {
        {
            std::lock_guard<std::mutex> lk(status_mtx_);
            active_name_ = name;
        }
        status_.store(status::loading, std::memory_order_release);
        std::string line = "{\"op\":\"load\",\"name\":\"" + name + "\"}";
        return write_line(line);
    }

    bool effect_runner::request_infer(const std::uint8_t* rgb, int w, int h)
    {
        if (rgb == nullptr || w <= 0 || h <= 0) return false;
        const long long n = static_cast<long long>(w) * h * 3;
        status_.store(status::running, std::memory_order_release);

        std::lock_guard<std::mutex> lk(write_mtx_);
        if (h_stdin_w_ == nullptr) return false;

        std::string header = "{\"op\":\"infer\",\"w\":" + std::to_string(w)
                           + ",\"h\":" + std::to_string(h)
                           + ",\"bytes\":" + std::to_string(n) + "}\n";
        DWORD written = 0;
        if (!WriteFile(h_stdin_w_, header.data(),
                       static_cast<DWORD>(header.size()),
                       &written, nullptr) || written != header.size())
        {
            std::lock_guard<std::mutex> lk2(status_mtx_);
            last_error_ = "WriteFile(infer header) failed";
            return false;
        }

        const std::uint8_t* p = rgb;
        std::size_t left = static_cast<std::size_t>(n);
        while (left > 0)
        {
            DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(left, 1 << 20));
            DWORD wrote = 0;
            if (!WriteFile(h_stdin_w_, p, chunk, &wrote, nullptr) || wrote == 0)
            {
                std::lock_guard<std::mutex> lk2(status_mtx_);
                last_error_ = "WriteFile(infer payload) failed";
                return false;
            }
            p    += wrote;
            left -= wrote;
        }
        return true;
    }

    bool effect_runner::request_cancel()
    {
        return write_line("{\"op\":\"cancel\"}");
    }

    bool effect_runner::request_unload()
    {
        return write_line("{\"op\":\"unload\"}");
    }

    bool effect_runner::restart()
    {
        std::string path;
        {
            std::lock_guard<std::mutex> lk(status_mtx_);
            path = script_path_;
        }
        if (path.empty()) return false;

        shutdown();

        {
            std::lock_guard<std::mutex> lk(events_mtx_);
            events_.clear();
        }
        {
            std::lock_guard<std::mutex> lk(status_mtx_);
            active_name_.clear();
            cached_effects_.clear();
            last_error_.clear();
        }
        status_.store(status::idle, std::memory_order_release);
        LOG_INFO("py_effect") << "restart: respawning worker";
        return start(path);
    }

    void effect_runner::reader_loop()
    {
        HANDLE h = static_cast<HANDLE>(h_stdout_r_);
        std::string line;
        while (helper::pipe_read_line(h, line))
        {
            std::string type;
            if (!helper::json_extract_string(line, "type", type))
            {
                LOG_WARN("py_effect") << "skipping unparsable frame: " << line;
                continue;
            }

            event ev;
            if      (type == "hello")       ev.kind = event_kind::hello;
            else if (type == "list")        ev.kind = event_kind::list;
            else if (type == "loaded")      ev.kind = event_kind::loaded;
            else if (type == "load_error")  ev.kind = event_kind::load_error;
            else if (type == "partial")     ev.kind = event_kind::partial;
            else if (type == "done")        ev.kind = event_kind::done;
            else if (type == "cancelled")   ev.kind = event_kind::cancelled;
            else if (type == "error")       ev.kind = event_kind::error;
            else if (type == "bye")         ev.kind = event_kind::bye;
            else if (type == "status")      ev.kind = event_kind::status;
            else if (type == "unloaded")    ev.kind = event_kind::unloaded;
            else
            {
                ev.kind    = event_kind::error;
                ev.message = "unknown frame type: " + type;
            }

            helper::json_extract_string(line, "name", ev.name);

            if (ev.kind == event_kind::list)
            {
                helper::json_extract_string_array(line, "names", ev.effects);
            }

            if (ev.kind == event_kind::loaded)
            {
                helper::json_extract_string(line, "label", ev.message);
            }
            else if (ev.kind == event_kind::load_error ||
                     ev.kind == event_kind::error)
            {
                helper::json_extract_string(line, "msg", ev.message);
            }
            else if (ev.kind == event_kind::status)
            {
                std::string phase, detail;
                helper::json_extract_string(line, "phase", phase);
                helper::json_extract_string(line, "msg",   detail);
                if (!phase.empty() && !detail.empty())
                    ev.message = phase + " - " + detail;
                else if (!phase.empty())
                    ev.message = phase;
                else
                    ev.message = detail;
                ev.name = phase;
            }

            if (ev.kind == event_kind::partial || ev.kind == event_kind::status)
            {
                double prog = 0.0;
                if (helper::json_extract_double(line, "progress", prog))
                    ev.progress = static_cast<float>(prog);
            }

            if (ev.kind == event_kind::partial || ev.kind == event_kind::done)
            {
                long long nbytes = 0;
                if (!helper::json_extract_int(line, "bytes", nbytes) || nbytes <= 0)
                {
                    LOG_ERROR("py_effect") << "missing/bad bytes field: " << line;
                    std::lock_guard<std::mutex> lk(status_mtx_);
                    last_error_ = "protocol: missing 'bytes' in " + type;
                    status_.store(status::failed, std::memory_order_release);
                    break;
                }
                ev.bytes.resize(static_cast<std::size_t>(nbytes));
                if (!helper::pipe_read_exact(h, ev.bytes.data(), ev.bytes.size()))
                {
                    LOG_ERROR("py_effect") << "short read on " << type
                                           << " payload (" << nbytes << " bytes)";
                    std::lock_guard<std::mutex> lk(status_mtx_);
                    last_error_ = "short read on " + type + " payload";
                    status_.store(status::failed, std::memory_order_release);
                    break;
                }
            }

            switch (ev.kind)
            {
            case event_kind::loaded:
                break;
            case event_kind::load_error:
                status_.store(status::failed, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lk(status_mtx_);
                    last_error_ = ev.message;
                }
                break;
            case event_kind::done:
                status_.store(status::ok, std::memory_order_release);
                break;
            case event_kind::error:
                status_.store(status::failed, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lk(status_mtx_);
                    last_error_ = ev.message;
                }
                break;
            case event_kind::cancelled:
                status_.store(status::idle, std::memory_order_release);
                break;
            case event_kind::unloaded:
                status_.store(status::idle, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lk(status_mtx_);
                    active_name_.clear();
                }
                break;
            default:
                break;
            }

            {
                std::lock_guard<std::mutex> lk(events_mtx_);
                const bool coalescable =
                    (ev.kind == event_kind::partial ||
                     ev.kind == event_kind::status);
                if (coalescable &&
                    !events_.empty() &&
                    events_.back().kind == ev.kind)
                {
                    events_.back() = std::move(ev);
                }
                else
                {
                    events_.push_back(std::move(ev));
                }
            }

            if (stopping_.load()) break;
        }

        alive_.store(false);
        if (!stopping_.load())
        {
            std::lock_guard<std::mutex> lk(status_mtx_);
            if (last_error_.empty())
                last_error_ = "python worker exited unexpectedly";
            status_.store(status::failed, std::memory_order_release);
        }
    }

    void effect_runner::poll(std::vector<event>& out, std::size_t max_events)
    {
        out.clear();
        std::lock_guard<std::mutex> lk(events_mtx_);
        while (!events_.empty() && out.size() < max_events)
        {
            out.push_back(std::move(events_.front()));
            events_.pop_front();
        }

        for (const event& ev : out)
        {
            if (ev.kind == event_kind::list)
            {
                std::lock_guard<std::mutex> lk2(status_mtx_);
                cached_effects_ = ev.effects;
            }
        }
    }

    std::string effect_runner::active_effect_name()
    {
        std::lock_guard<std::mutex> lk(status_mtx_);
        return active_name_;
    }

    std::string effect_runner::last_error()
    {
        std::lock_guard<std::mutex> lk(status_mtx_);
        return last_error_;
    }

    std::vector<std::string> effect_runner::cached_effects()
    {
        std::lock_guard<std::mutex> lk(status_mtx_);
        return cached_effects_;
    }

    void effect_runner::set_cached_effects(std::vector<std::string> names)
    {
        std::lock_guard<std::mutex> lk(status_mtx_);
        cached_effects_ = std::move(names);
    }

    std::string effect_runner::picked_python()
    {
        std::lock_guard<std::mutex> lk(status_mtx_);
        return picked_python_;
    }

    std::vector<std::string>
    effect_runner::recent_log_lines(std::size_t max_lines)
    {
        std::lock_guard<std::mutex> lk(log_mtx_);
        std::vector<std::string> out;
        if (log_lines_.empty() || max_lines == 0) return out;
        const std::size_t take = std::min(max_lines, log_lines_.size());
        out.reserve(take);
        auto it = log_lines_.end() - static_cast<std::ptrdiff_t>(take);
        for (; it != log_lines_.end(); ++it) out.push_back(*it);
        return out;
    }

    void effect_runner::stderr_loop()
    {
        HANDLE h = static_cast<HANDLE>(h_stderr_r_);
        std::string buf;
        buf.reserve(1024);

        auto push_line = [this](std::string line)
        {
            if (line.empty()) return;
            LOG_INFO("py_worker") << line;
            std::lock_guard<std::mutex> lk(log_mtx_);
            log_lines_.emplace_back(std::move(line));
            while (log_lines_.size() > k_log_capacity) log_lines_.pop_front();
        };

        char chunk[512];
        while (true)
        {
            DWORD got = 0;
            BOOL ok = ReadFile(h, chunk, sizeof(chunk), &got, nullptr);
            if (!ok || got == 0) break;
            for (DWORD i = 0; i < got; ++i)
            {
                const char c = chunk[i];
                if (c == '\n' || c == '\r')
                {
                    push_line(std::move(buf));
                    buf.clear();
                }
                else
                {
                    buf.push_back(c);
                }
            }
            if (stopping_.load()) break;
        }
        if (!buf.empty()) push_line(std::move(buf));
    }
} // namespace fox_tracer
