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
#ifndef RAYTRACER_WITH_AI_PYTHON_ENV_H
#define RAYTRACER_WITH_AI_PYTHON_ENV_H

#include <atomic>
#include <mutex>
#include <string>

namespace fox_tracer::utility
{
    class python_env
    {
    public:
        python_env() noexcept = default;
        ~python_env() = default;

        python_env(const python_env&)            = delete;
        python_env& operator=(const python_env&) = delete;

        [[nodiscard]] std::string resolve_venv_python() const;

        void create_venv();
        void install_ai_deps_now(const std::string &index_url,
                                 const std::string& worker_python);

        [[nodiscard]] bool cuda_available() const;
        void refresh_cuda_detection() const;

        [[nodiscard]] bool is_busy() const noexcept
        { return busy_.load(std::memory_order_relaxed); }

        bool consume_pending_restart() noexcept
        { return pending_worker_restart_.exchange(false); }

        [[nodiscard]] const std::atomic<bool>* busy_flag() const noexcept
        {
            return &busy_;
        }

        std::mutex&        last_summary_mutex   ()       noexcept { return last_summary_mtx_; }
        const std::string& last_summary_unlocked() const noexcept { return last_summary_; }

    private:
        void run_async(std::string command_line,
                       std::string log_tag,
                       bool restart_worker_on_success);

        std::atomic<bool> busy_{false};
        std::atomic<bool> pending_worker_restart_{false};

        std::mutex  last_summary_mtx_;
        std::string last_summary_;

        mutable bool cuda_detected_cached_{false};
        mutable bool cuda_detection_done_{false};
    };
} // namespace fox_tracer
#endif //RAYTRACER_WITH_AI_PYTHON_ENV_H
