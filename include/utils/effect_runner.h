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
#ifndef RAYTRACER_WITH_AI_EFFECT_RUNNER_H
#define RAYTRACER_WITH_AI_EFFECT_RUNNER_H


#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fox_tracer::utility
{
    class effect_runner
    {
    public:
        enum class status : int
        {
            idle    = 0,
            loading = 1,
            running = 2,
            ok      = 3,
            failed  = 4
        };

        enum class event_kind : int
        {
            hello,
            list,
            loaded,
            load_error,
            partial,
            done,
            cancelled,
            error,
            bye,
            status,
            unloaded
        };

        struct event
        {
            event_kind                     kind{event_kind::hello};
            std::string                    name;
            std::string                    message;
            float                          progress{0.0f};
            int                            width{0};
            int                            height{0};
            std::vector<std::uint8_t>      bytes;
            std::vector<std::string>       effects;
        };

        effect_runner()  noexcept;
        ~effect_runner();

        effect_runner(const effect_runner&)            = delete;
        effect_runner& operator=(const effect_runner&) = delete;
        effect_runner(effect_runner&&)                 = delete;
        effect_runner& operator=(effect_runner&&)      = delete;

        bool start(const std::string& script_path);

        bool restart ();
        void shutdown();

        [[nodiscard]] bool alive() const noexcept { return alive_.load(); }

        bool request_load (const std::string& name);
        bool request_infer(const std::uint8_t* rgb, int w, int h);
        bool request_cancel ();
        bool request_unload ();
        bool request_list   ();

        void poll(std::vector<event>& out, std::size_t max_events = 64);

        [[nodiscard]] status current_status() const noexcept
        {
            return status_.load(std::memory_order_acquire);
        }

        [[nodiscard]] std::string               active_effect_name  ();
        [[nodiscard]] std::string               last_error          ();
        [[nodiscard]] std::vector<std::string>  cached_effects      ();
        [[nodiscard]] std::string               picked_python       ();

        [[nodiscard]] std::vector<std::string>
        recent_log_lines(std::size_t max_lines = 200);

        void set_cached_effects(std::vector<std::string> names);

    private:
        void reader_loop();
        void stderr_loop();
        void force_kill ();

        bool write_line(const std::string& line);
        bool write_bytes(const std::uint8_t* p, std::size_t n);

        void* h_process_ {nullptr};
        void* h_stdin_w_ {nullptr};
        void* h_stdout_r_{nullptr};
        void* h_stderr_r_{nullptr};

        std::thread         reader_;
        std::thread         stderr_reader_;
        std::atomic<bool>   alive_{false};
        std::atomic<bool>   stopping_{false};

        std::mutex              log_mtx_;
        std::deque<std::string> log_lines_;

        static constexpr std::size_t k_log_capacity = 400;

        std::atomic<status> status_{status::idle};

        std::mutex        events_mtx_;
        std::deque<event> events_;
        std::mutex        write_mtx_;

        std::mutex           status_mtx_;
        std::string          active_name_;
        std::string          last_error_;
        std::vector<std::string> cached_effects_;
        std::string          script_path_;
        std::string          picked_python_;
    };
} // namespace fox_tracer::utility

#endif //RAYTRACER_WITH_AI_EFFECT_RUNNER_H
