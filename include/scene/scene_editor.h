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
#ifndef RAYTRACER_WITH_AI_SCENE_EDITOR_H
#define RAYTRACER_WITH_AI_SCENE_EDITOR_H


#include "utils/assets_loader.h"

#include <string>
#include <vector>

namespace fox_tracer::scene
{
    class container;

    struct inst_trs
    {
        float pos[3]   {0.0f, 0.0f, 0.0f};
        float rot[3]   {0.0f, 0.0f, 0.0f};
        float scale[3] {1.0f, 1.0f, 1.0f};
    };

    class scene_editor
    {
    public:
        std::string assets_root  {"assets"};
        std::string scenes_subdir{"scene"};
        std::string saved_subdir {"saved"};

        int target_width {0};
        int target_height{0};

        std::string current_scene_name;
        loader::gem_scene   gem_state;

        std::vector<inst_trs> inst_edits;

        bool        pending_rebuild  {false};
        bool        pending_load     {false};
        std::string pending_load_path;

        bool show_editor      {false};
        int  selected_instance{-1};

        bool open_new_scene_modal{false};
        char new_scene_name_buf[128] {'m','y','-','s','c','e','n','e','\0'};
        std::string last_save_message;

        char  add_mesh_buf       [256] {'m','e','s','h','.','o','b','j','\0'};
        float add_pos            [3]   {0.0f, 0.0f, 0.0f};
        float add_rot            [3]   {0.0f, 0.0f, 0.0f};
        float add_scale          [3]   {1.0f, 1.0f, 1.0f};
        int   add_bsdf_idx       {0};
        char  add_reflectance_buf[256] {0};
        float add_colour         [3]   {0.8f, 0.8f, 0.8f};
        float add_emission       [3]   {0.0f, 0.0f, 0.0f};
        float add_roughness      {0.3f};
        float add_int_ior        {1.5f};

        std::vector<std::string> available_scenes;
        std::vector<std::string> available_textures;

        void       scan_scenes       ();
        void       scan_textures     ();
        void       seed_empty_scene  ();
        bool       save_current_scene();
        void       request_rebuild   () noexcept;
        container* apply_pending     ();

        void bind_to            (const std::string& scene_folder);
        bool create_empty_scene (const std::string& name);
        void request_load       (const std::string& folder);


        void draw_scene_menus ();
        void draw_scene_popups();
        void draw_scene_status() const;
        void draw_editor      ();

    private:
        void rebuild_inst_edits_from_matrices   ();
        void draw_scene_settings                ();
        void draw_environment                   ();
        void draw_object_list                   ();
        void draw_selected_object               ();
        void draw_add_object                    ();

        void compose_matrix_for(int i);

    };
} // namespace fox_tracer

#endif //RAYTRACER_WITH_AI_SCENE_EDITOR_H
