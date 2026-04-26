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
#ifndef RAYTRACER_WITH_AI_HELPER_H
#define RAYTRACER_WITH_AI_HELPER_H


#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fox_tracer::helper
{
    std::string json_escape (const std::string& s);
    std::string vec3_str    (float x, float y, float z);
    std::string float_str   (float v);
    std::string fmt_label   (const std::string& prefix, int counter);

    bool ends_with_ci    (const std::string& s, const std::string& suffix);
    std::string leaf_name(const std::string& p);

    int clamp_tile_size  (int v);
    float power_heuristic(float pdf_a, float pdf_b) noexcept;

    bool has_scene_json (const std::string& dir);
    void list_subdirs   (const std::string& dir, std::vector<std::string>& out);
    void list_files     (const std::string& dir, const std::vector<std::string>& extensions, std::vector<std::string>& out);


    std::string unescape_json(std::string_view in);
    std::size_t json_find_key(std::string_view src, std::string_view key);

    bool json_extract_string      (std::string_view src, std::string_view key, std::string& out);
    bool json_extract_int         (std::string_view src, std::string_view key, long long& out);
    bool json_extract_double      (std::string_view src, std::string_view key, double& out);
    bool json_extract_string_array(std::string_view src, std::string_view key, std::vector<std::string>& out);

    bool pipe_read_exact(void* handle, std::uint8_t* p, std::size_t n);
    bool pipe_read_line (void* handle, std::string& out);


    std::string pick_python              (const std::string& script_path);
    std::string build_python_command_line(const std::string& python_exe,
                                          const std::string& script_path);

    std::string find_venv_base_python();
} // namespace fox_tracer::helper

#endif //RAYTRACER_WITH_AI_HELPER_H
