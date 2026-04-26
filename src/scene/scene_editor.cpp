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

#include "scene/scene_editor.h"
#include "scene/scene.h"
#include "scene/scene_loader.h"

#include "framework/core.h"
#include "framework/base.h"
#include "utils/helper.h"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace fox_tracer::scene
{
    namespace
    {
        void set_prop(std::vector<loader::gem_property>& props,
                      const std::string& name, const std::string& value)
        {
            for (auto& p : props)
            {
                if (p.name == name) { p.value = value; return; }
            }
            loader::gem_property p; p.name = name; p.value = value;
            props.push_back(p);
        }

        void compose_trs(const float t[3], const float r_deg[3],
                         const float s[3], loader::gem_matrix& out)
        {
            const matrix T  = transform::translation(vec3(t[0], t[1], t[2]));
            const matrix Rx = transform::rotate_x(r_deg[0] * math::pi<float> / 180.0f);
            const matrix Ry = transform::rotate_y(r_deg[1] * math::pi<float> / 180.0f);
            const matrix Rz = transform::rotate_z(r_deg[2] * math::pi<float> / 180.0f);
            const matrix S  = transform::scaling(vec3(s[0], s[1], s[2]));

            matrix M = T;
            M = M * Ry;
            M = M * Rx;
            M = M * Rz;
            M = M * S;
            for (int i = 0; i < 16; ++i) out.m[i] = M.m[i];
        }

        void decompose_trs(const loader::gem_matrix& in, inst_trs& out)
        {
            out.pos[0] = in.m[3];
            out.pos[1] = in.m[7];
            out.pos[2] = in.m[11];

            const float r00 = in.m[0], r01 = in.m[1], r02 = in.m[2];
            const float r10 = in.m[4], r11 = in.m[5], r12 = in.m[6];
            const float r20 = in.m[8], r21 = in.m[9], r22 = in.m[10];

            const float sx = std::sqrt(r00 * r00 + r10 * r10 + r20 * r20);
            const float sy = std::sqrt(r01 * r01 + r11 * r11 + r21 * r21);
            const float sz = std::sqrt(r02 * r02 + r12 * r12 + r22 * r22);

            out.scale[0] = sx > 0.0f ? sx : 1.0f;
            out.scale[1] = sy > 0.0f ? sy : 1.0f;
            out.scale[2] = sz > 0.0f ? sz : 1.0f;

            const float inv_sx = 1.0f / out.scale[0];
            const float inv_sy = 1.0f / out.scale[1];
            const float inv_sz = 1.0f / out.scale[2];

            const float m00 = r00 * inv_sx, m01 = r01 * inv_sy, m02 = r02 * inv_sz;
            const float m10 = r10 * inv_sx, m11 = r11 * inv_sy, m12 = r12 * inv_sz;
            const float m20 = r20 * inv_sx, m21 = r21 * inv_sy, m22 = r22 * inv_sz;

            const float rx = std::asin(std::clamp(-m21, -1.0f, 1.0f));
            float ry, rz;
            if (std::cos(rx) > 1e-4f)
            {
                ry = std::atan2(m20, m22);
                rz = std::atan2(m01, m11);
            }
            else
            {
                ry = std::atan2(-m02, m00);
                rz = 0.0f;
            }
            constexpr float rad_to_deg = 180.0f / math::pi<float>;
            out.rot[0] = rx * rad_to_deg;
            out.rot[1] = ry * rad_to_deg;
            out.rot[2] = rz * rad_to_deg;
        }
    } // namespace

    void scene_editor::scan_scenes()
    {
        available_scenes.clear();

        const std::string scenes_root = assets_root + "/" + scenes_subdir;

        std::vector<std::string> roots_to_scan = { scenes_root, assets_root };
        for (const auto& root : roots_to_scan)
        {
            std::vector<std::string> level1;
            helper::list_subdirs(root, level1);
            for (const auto& sub : level1)
            {
                if (helper::has_scene_json(sub))
                {
                    available_scenes.push_back(sub);
                }
                else
                {
                    std::vector<std::string> level2;
                    helper::list_subdirs(sub, level2);
                    for (const auto& sub2 : level2)
                    {
                        if (helper::has_scene_json(sub2)) available_scenes.push_back(sub2);
                    }
                }
            }
        }
        std::sort(available_scenes.begin(), available_scenes.end());
        available_scenes.erase(
            std::unique(available_scenes.begin(), available_scenes.end()),
            available_scenes.end());
    }

    void scene_editor::scan_textures()
    {
        available_textures.clear();
        const std::string textures_root = assets_root + "/textures";
        helper::list_files(textures_root,
                   { ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".hdr" },
                   available_textures);
        std::sort(available_textures.begin(), available_textures.end());
    }

    void scene_editor::bind_to(const std::string& scene_folder)
    {
        current_scene_name = scene_folder;
        gem_state = loader::gem_scene();
        gem_state.load(scene_folder + "/scene.json");
        selected_instance = -1;
        rebuild_inst_edits_from_matrices();
        scan_textures();
    }

    void scene_editor::rebuild_inst_edits_from_matrices()
    {
        inst_edits.clear();
        inst_edits.resize(gem_state.instances.size());
        for (size_t i = 0; i < gem_state.instances.size(); ++i)
        {
            decompose_trs(gem_state.instances[i].w, inst_edits[i]);
        }
    }

    void scene_editor::compose_matrix_for(int i)
    {
        if (i < 0 || i >= static_cast<int>(gem_state.instances.size())) return;
        compose_trs(inst_edits[i].pos, inst_edits[i].rot, inst_edits[i].scale,
                    gem_state.instances[i].w);
    }

    void scene_editor::seed_empty_scene()
    {
        gem_state = loader::gem_scene();
        inst_edits.clear();
        const int w = target_width  > 0 ? target_width  : 1024;
        const int h = target_height > 0 ? target_height : 1024;
        set_prop(gem_state.sceneProperties, "width",  std::to_string(w));
        set_prop(gem_state.sceneProperties, "height", std::to_string(h));
        set_prop(gem_state.sceneProperties, "fov",    "45");
        set_prop(gem_state.sceneProperties, "from",   "0 1 5");
        set_prop(gem_state.sceneProperties, "to",     "0 1 0");
        set_prop(gem_state.sceneProperties, "up",     "0 1 0");
        selected_instance = -1;
    }

    bool scene_editor::create_empty_scene(const std::string& name)
    {
        if (name.empty())
        {
            last_save_message = "New Scene: empty name";
            return false;
        }
        const std::string scenes_root = assets_root + "/" + scenes_subdir;
        CreateDirectoryA(scenes_root.c_str(), nullptr);

        const std::string folder = scenes_root + "/" + name;
        const BOOL ok = CreateDirectoryA(folder.c_str(), nullptr);
        if (!ok && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            last_save_message = "New Scene: cannot create " + folder;
            return false;
        }

        current_scene_name = folder;
        seed_empty_scene();
        if (!save_current_scene()) return false;

        last_save_message = "Created " + folder;
        pending_rebuild = true;
        scan_scenes();
        return true;
    }

    bool scene_editor::save_current_scene()
    {
        if (current_scene_name.empty())
        {
            last_save_message = "Save: no current scene";
            return false;
        }
        const std::string path = current_scene_name + "/scene.json";
        std::ofstream f(path);
        if (!f.is_open())
        {
            last_save_message = "Save: cannot open " + path;
            return false;
        }
        f << "{\n";
        for (const auto & p : gem_state.sceneProperties)
        {
            f << "    \"" << helper::json_escape(p.name) << "\": \""
              << helper::json_escape(p.value) << "\",\n";
        }
        f << "    \"instances\": [";
        for (size_t i = 0; i < gem_state.instances.size(); ++i)
        {
            const auto& inst = gem_state.instances[i];
            f << (i == 0 ? "\n        {" : ", {");
            f << "\n            \"filename\": \""
              << helper::json_escape(inst.meshFilename) << "\",";
            f << "\n            \"world\": [";
            for (int k = 0; k < 16; ++k)
            {
                if (k) f << ", ";
                f << inst.w.m[k];
            }
            f << "]";
            for (const auto& p : inst.material.properties)
            {
                f << ",\n            \"" << helper::json_escape(p.name) << "\": \""
                  << helper::json_escape(p.value) << "\"";
            }
            f << "\n        }";
        }
        f << (gem_state.instances.empty() ? "]\n" : "\n    ]\n");
        f << "}\n";
        f.close();
        last_save_message = "Saved " + path;
        return true;
    }

    void scene_editor::request_load(const std::string& folder)
    {
        pending_load_path = folder;
        pending_load = true;
    }

    void scene_editor::request_rebuild() noexcept
    {
        pending_rebuild = true;
    }

    scene::container* scene_editor::apply_pending()
    {
        if (pending_load)
        {
            bind_to(pending_load_path);
            pending_load = false;
            pending_load_path.clear();
            pending_rebuild = true;
        }
        if (!pending_rebuild) return nullptr;
        pending_rebuild = false;
        return build_scene_from_gem(gem_state, current_scene_name,
                                    target_width, target_height);
    }

    void scene_editor::draw_scene_menus()
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Rescan assets/"))
            {
                scan_scenes();
                scan_textures();
            }
            if (ImGui::BeginMenu("Load Scene", !available_scenes.empty()))
            {
                for (const auto& path : available_scenes)
                {
                    const bool current = (path == current_scene_name);
                    const std::string label = helper::leaf_name(path);
                    if (ImGui::MenuItem(label.c_str(), nullptr, current))
                    {
                        request_load(path);
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Reload Current Scene", nullptr, false,
                                !current_scene_name.empty()))
            {
                request_load(current_scene_name);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene"))
        {
            if (ImGui::MenuItem("New Empty Scene..."))
            {
                open_new_scene_modal = true;
            }
            if (ImGui::MenuItem("Save Scene", nullptr, false,
                                !current_scene_name.empty()))
            {
                save_current_scene();
                pending_rebuild = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Scene Editor", nullptr, show_editor))
            {
                show_editor = !show_editor;
            }
            if (ImGui::MenuItem("Rebuild Now", nullptr, false,
                                !current_scene_name.empty()))
            {
                request_rebuild();
            }
            ImGui::EndMenu();
        }
    }

    void scene_editor::draw_scene_popups()
    {
        if (open_new_scene_modal)
        {
            ImGui::OpenPopup("New Empty Scene");
            open_new_scene_modal = false;
        }
        ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f));
        if (ImGui::BeginPopupModal("New Empty Scene", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::TextWrapped("Creates %s/%s/<name>/scene.json and loads it.",
                               assets_root.c_str(), scenes_subdir.c_str());
            ImGui::InputText("Name", new_scene_name_buf, sizeof(new_scene_name_buf));
            ImGui::Separator();
            if (ImGui::Button("Create", ImVec2(120, 0)))
            {
                if (create_empty_scene(new_scene_name_buf))
                {
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0)))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    void scene_editor::draw_scene_status() const
    {
        if (!current_scene_name.empty())
        {
            ImGui::TextDisabled("[%s]", helper::leaf_name(current_scene_name).c_str());
        }
        if (!last_save_message.empty())
        {
            if (!current_scene_name.empty()) ImGui::Separator();
            ImGui::TextDisabled("%s", last_save_message.c_str());
        }
    }

    void scene_editor::draw_editor()
    {
        if (!show_editor) return;

        ImGui::SetNextWindowPos (ImVec2(380.0f, 30.0f),  ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420.0f, 620.0f), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Scene Editor", &show_editor))
        {
            ImGui::End();
            return;
        }

        ImGui::TextWrapped("Scene: %s", helper::leaf_name(current_scene_name).c_str());
        ImGui::Separator();

        draw_scene_settings();
        ImGui::Separator();
        draw_environment();
        ImGui::Separator();
        draw_object_list();
        ImGui::Separator();
        draw_selected_object();
        ImGui::Separator();
        draw_add_object();
        ImGui::Separator();

        if (ImGui::Button("Apply / Rebuild Scene"))
        {
            request_rebuild();
        }
        ImGui::SameLine();
        if (ImGui::Button("Revert (reload from disk)"))
        {
            request_load(current_scene_name);
        }
        ImGui::SameLine();
        if (ImGui::Button("Save to Disk"))
        {
            save_current_scene();
        }

        ImGui::End();
    }

    void scene_editor::draw_scene_settings()
    {
        if (!ImGui::CollapsingHeader("Scene Settings",
                                     ImGuiTreeNodeFlags_DefaultOpen)) return;

        float fov  = gem_state.findProperty("fov").getValue(45.0f);
        float from[3], to[3], up[3];
        gem_state.findProperty("from").getValuesAsVector3(from[0], from[1], from[2]);
        gem_state.findProperty("to")  .getValuesAsVector3(to[0],   to[1],   to[2]);
        gem_state.findProperty("up")  .getValuesAsVector3(up[0],   up[1],   up[2]);

        bool changed = false;
        changed |= ImGui::SliderFloat ("FOV",  &fov, 10.0f, 120.0f);
        changed |= ImGui::InputFloat3("From", from);
        changed |= ImGui::InputFloat3("To",   to);
        changed |= ImGui::InputFloat3("Up",   up);
        if (changed)
        {
            set_prop(gem_state.sceneProperties, "fov",  helper::float_str(fov));
            set_prop(gem_state.sceneProperties, "from", helper::vec3_str(from[0], from[1], from[2]));
            set_prop(gem_state.sceneProperties, "to",   helper::vec3_str(to[0],   to[1],   to[2]));
            set_prop(gem_state.sceneProperties, "up",   helper::vec3_str(up[0],   up[1],   up[2]));
        }
    }

    void scene_editor::draw_environment()
    {
        if (!ImGui::CollapsingHeader("Environment")) return;

        const std::string envmap = gem_state.findProperty("envmap")
                                            .getValue(std::string{});
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s", envmap.c_str());
        if (ImGui::InputText("Envmap (assets/textures)", buf, sizeof(buf)))
        {
            set_prop(gem_state.sceneProperties, "envmap", buf);
        }

        if (ImGui::BeginCombo("Pick envmap##env", "browse..."))
        {
            for (const auto& t : available_textures)
            {
                if (ImGui::Selectable(t.c_str()))
                {
                    set_prop(gem_state.sceneProperties, "envmap", t);
                }
            }
            ImGui::EndCombo();
        }
        ImGui::TextDisabled("Leave envmap empty for a solid background.");
    }

    void scene_editor::draw_object_list()
    {
        ImGui::Text("Objects (%d):", static_cast<int>(gem_state.instances.size()));
        ImGui::BeginChild("##objs", ImVec2(0, 120), true);
        for (int i = 0; i < static_cast<int>(gem_state.instances.size()); ++i)
        {
            auto& inst = gem_state.instances[i];
            char label[320];
            std::snprintf(label, sizeof(label), "[%d] %s##o%d",
                          i, inst.meshFilename.c_str(), i);
            if (ImGui::Selectable(label, selected_instance == i))
            {
                selected_instance = i;
            }
        }
        ImGui::EndChild();
        if (ImGui::Button("Delete Selected")
            && selected_instance >= 0
            && selected_instance < static_cast<int>(gem_state.instances.size()))
        {
            gem_state.instances.erase(gem_state.instances.begin() + selected_instance);
            if (selected_instance < static_cast<int>(inst_edits.size()))
            {
                inst_edits.erase(inst_edits.begin() + selected_instance);
            }
            selected_instance = -1;
        }
    }

    void scene_editor::draw_selected_object()
    {
        ImGui::PushID("selected_object");
        struct scope_pop_id { ~scope_pop_id() { ImGui::PopID(); } } _pop{};

        if (selected_instance < 0
            || selected_instance >= static_cast<int>(gem_state.instances.size()))
        {
            ImGui::TextDisabled("No object selected.");
            return;
        }
        if (inst_edits.size() != gem_state.instances.size())
        {
            rebuild_inst_edits_from_matrices();
        }

        auto& inst = gem_state.instances[selected_instance];
        auto& edit = inst_edits[selected_instance];

        char mesh_buf[256];
        std::snprintf(mesh_buf, sizeof(mesh_buf), "%s", inst.meshFilename.c_str());
        if (ImGui::InputText("Mesh file", mesh_buf, sizeof(mesh_buf)))
        {
            inst.meshFilename = mesh_buf;
        }

        bool trs_changed = false;
        trs_changed |= ImGui::DragFloat3("Position",     edit.pos,   0.01f);
        trs_changed |= ImGui::DragFloat3("Rotate (deg)", edit.rot,   1.0f);
        trs_changed |= ImGui::DragFloat3("Scale",        edit.scale, 0.01f, 0.001f, 1000.0f);
        if (trs_changed)
        {
            compose_matrix_for(selected_instance);
        }

        ImGui::Separator();
        ImGui::Text("Material");

        const std::string bsdf_name = inst.material.find("bsdf").getValue(std::string{});
        const char* bsdfs[] = { "diffuse", "orennayar", "mirror", "glass",
                                "plastic", "dielectric", "conductor" };
        int bsdf_idx = 0;
        for (int i = 0; i < static_cast<int>(IM_ARRAYSIZE(bsdfs)); ++i)
        {
            if (bsdfs[i] == bsdf_name) bsdf_idx = i;
        }
        if (ImGui::Combo("BSDF", &bsdf_idx, bsdfs, IM_ARRAYSIZE(bsdfs)))
        {
            set_prop(inst.material.properties, "bsdf", bsdfs[bsdf_idx]);
        }

        const std::string refl = inst.material.find("reflectance").getValue(std::string{});
        char rbuf[256];
        std::snprintf(rbuf, sizeof(rbuf), "%s", refl.c_str());
        if (ImGui::InputText("Reflectance (file)", rbuf, sizeof(rbuf)))
        {
            set_prop(inst.material.properties, "reflectance", rbuf);
        }

        if (ImGui::BeginCombo("Pick texture##sel",
                              refl.empty() ? "browse..." : refl.c_str()))
        {
            if (ImGui::Selectable("<none>"))
            {
                set_prop(inst.material.properties, "reflectance", "");
            }
            for (const auto& t : available_textures)
            {
                if (ImGui::Selectable(t.c_str()))
                {
                    set_prop(inst.material.properties, "reflectance", t);
                }
            }
            ImGui::EndCombo();
        }

        float rough = inst.material.find("roughness").getValue(0.3f);
        if (ImGui::SliderFloat("Roughness", &rough, 0.001f, 1.0f))
        {
            set_prop(inst.material.properties, "roughness", helper::float_str(rough));
        }
        float int_ior = inst.material.find("intIOR").getValue(1.5f);
        if (ImGui::SliderFloat("IOR", &int_ior, 1.0f, 3.0f))
        {
            set_prop(inst.material.properties, "intIOR", helper::float_str(int_ior));
        }

        ImGui::Separator();
        ImGui::Text("UV Transform");
        float uv_rot = inst.material.find("uv_rot").getValue(0.0f);
        if (ImGui::SliderFloat("Texture Rotation (deg)", &uv_rot, -180.0f, 180.0f))
        {
            set_prop(inst.material.properties, "uv_rot", helper::float_str(uv_rot));
        }
        float uv_scale[2] = { 1.0f, 1.0f };
        float tmp = 0.0f;
        inst.material.find("uv_scale").getValuesAsVector3(uv_scale[0], uv_scale[1], tmp);
        if (uv_scale[0] == 0.0f) uv_scale[0] = 1.0f;
        if (uv_scale[1] == 0.0f) uv_scale[1] = 1.0f;
        if (ImGui::DragFloat2("Texture Scale", uv_scale, 0.01f, 0.01f, 100.0f))
        {
            set_prop(inst.material.properties, "uv_scale",
                     helper::vec3_str(uv_scale[0], uv_scale[1], 0.0f));
        }
        float uv_offset[2] = { 0.0f, 0.0f };
        inst.material.find("uv_offset").getValuesAsVector3(uv_offset[0], uv_offset[1], tmp);
        if (ImGui::DragFloat2("Texture Offset", uv_offset, 0.01f))
        {
            set_prop(inst.material.properties, "uv_offset",
                     helper::vec3_str(uv_offset[0], uv_offset[1], 0.0f));
        }
        ImGui::Separator();

        float e[3] = { 0.0f, 0.0f, 0.0f };
        inst.material.find("emission").getValuesAsVector3(e[0], e[1], e[2]);
        bool has_emission = inst.material.find("emission").getValue(std::string{}) != "";
        if (ImGui::Checkbox("Emits Light", &has_emission))
        {
            if (has_emission)
            {
                set_prop(inst.material.properties, "emission",
                         helper::vec3_str(e[0] > 0.0f ? e[0] : 1.0f,
                                  e[1] > 0.0f ? e[1] : 1.0f,
                                  e[2] > 0.0f ? e[2] : 1.0f));
            }
            else
            {
                set_prop(inst.material.properties, "emission", "");
            }
        }
        if (has_emission)
        {
            if (ImGui::DragFloat3("Emission", e, 0.1f, 0.0f, 1000.0f))
            {
                set_prop(inst.material.properties, "emission",
                         helper::vec3_str(e[0], e[1], e[2]));
            }
        }
    }

    void scene_editor::draw_add_object()
    {
        if (!ImGui::CollapsingHeader("Add Object")) return;

        ImGui::PushID("add_object");
        struct scope_pop_id { ~scope_pop_id() { ImGui::PopID(); } } _pop{};

        ImGui::InputText("Mesh file", add_mesh_buf, sizeof(add_mesh_buf));
        ImGui::TextDisabled(".gem (binary) or .obj (text) "
                            "-- path relative to scene folder");
        ImGui::DragFloat3("Translate",    add_pos,   0.01f);
        ImGui::DragFloat3("Rotate (deg)", add_rot,   1.0f);
        ImGui::DragFloat3("Scale",        add_scale, 0.01f, 0.001f, 100.0f);

        const char* bsdfs[] = { "diffuse", "orennayar", "mirror", "glass",
                                "plastic", "dielectric", "conductor" };
        ImGui::Combo("BSDF##add", &add_bsdf_idx, bsdfs, IM_ARRAYSIZE(bsdfs));
        ImGui::InputText("Reflectance (file)##add", add_reflectance_buf,
                         sizeof(add_reflectance_buf));
        if (ImGui::BeginCombo("Pick texture##add",
                              add_reflectance_buf[0] ? add_reflectance_buf : "browse..."))
        {
            if (ImGui::Selectable("<none>"))
            {
                add_reflectance_buf[0] = '\0';
            }
            for (const auto& t : available_textures)
            {
                if (ImGui::Selectable(t.c_str()))
                {
                    std::snprintf(add_reflectance_buf, sizeof(add_reflectance_buf),
                                  "%s", t.c_str());
                }
            }
            ImGui::EndCombo();
        }
        ImGui::ColorEdit3("Tint (if no texture)", add_colour);
        ImGui::SliderFloat("Roughness##add", &add_roughness, 0.001f, 1.0f);
        ImGui::SliderFloat("IOR##add",       &add_int_ior,   1.0f,   3.0f);
        ImGui::DragFloat3 ("Emission (0 = off)", add_emission, 0.1f, 0.0f, 1000.0f);

        auto push_inst = [&](loader::gem_instance&& inst)
        {
            gem_state.instances.push_back(std::move(inst));
            inst_trs trs;
            for (int k = 0; k < 3; ++k) trs.pos[k]   = add_pos[k];
            for (int k = 0; k < 3; ++k) trs.rot[k]   = add_rot[k];
            for (int k = 0; k < 3; ++k) trs.scale[k] = add_scale[k];
            inst_edits.push_back(trs);
            selected_instance = static_cast<int>(gem_state.instances.size()) - 1;
        };

        if (ImGui::Button("Add Mesh"))
        {
            loader::gem_instance inst;
            inst.meshFilename = add_mesh_buf;
            compose_trs(add_pos, add_rot, add_scale, inst.w);
            set_prop(inst.material.properties, "bsdf", bsdfs[add_bsdf_idx]);
            if (add_reflectance_buf[0])
            {
                set_prop(inst.material.properties, "reflectance",
                         add_reflectance_buf);
            }
            set_prop(inst.material.properties, "roughness", helper::float_str(add_roughness));
            set_prop(inst.material.properties, "intIOR",    helper::float_str(add_int_ior));
            if (add_emission[0] > 0.0f
                || add_emission[1] > 0.0f
                || add_emission[2] > 0.0f)
            {
                set_prop(inst.material.properties, "emission",
                         helper::vec3_str(add_emission[0],
                                  add_emission[1],
                                  add_emission[2]));
            }
            push_inst(std::move(inst));
        }
        ImGui::SameLine();
        if (ImGui::Button("Add Area Light (unit quad)"))
        {
            loader::gem_instance inst;
            inst.meshFilename = "Rectangle.gem";
            compose_trs(add_pos, add_rot, add_scale, inst.w);
            set_prop(inst.material.properties, "bsdf",        "diffuse");
            set_prop(inst.material.properties, "reflectance", "0_0_0_1.0.png");
            const float em[3] = {
                add_emission[0] > 0.0f ? add_emission[0] : 10.0f,
                add_emission[1] > 0.0f ? add_emission[1] : 10.0f,
                add_emission[2] > 0.0f ? add_emission[2] : 10.0f
            };
            set_prop(inst.material.properties, "emission",
                     helper::vec3_str(em[0], em[1], em[2]));
            push_inst(std::move(inst));
        }
    }
} // namespace fox_tracer::scene
