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

#include "scene/scene_host.h"
#include "scene/scene.h"
#include "scene/scene_loader.h"

#include "config.h"
#include "utils/logger.h"
#include "utils/paths.h"
#include "render/renderer.h"

namespace fox_tracer::scene
{
    scene_host::~scene_host()
    {
        release();
    }

    bool scene_host::init(const std::string& scene_name,
                          const std::string& assets_root,
                          int width, int height)
    {
        const std::string resolved_scene = paths::resolve(scene_name);
        LOG_INFO("engine") << "loading scene " << resolved_scene
                           << " (" << width << "x" << height << ")";

        const bool scene_path_ok = paths::exists(resolved_scene);
        if (!scene_path_ok)
        {
            LOG_ERROR("engine") << "scene directory not found: " << resolved_scene;
        }

        current_ = load_scene(resolved_scene, width, height);
        if (current_ == nullptr)
        {
            LOG_ERROR("engine") << "failed to load scene";
            return false;
        }

        if (!scene_path_ok)
        {
            config().pause_render.store(true, std::memory_order_relaxed);
            scene_idle_until_loaded_ = true;
        }

        editor_.assets_root   = paths::resolve(assets_root);
        editor_.target_width  = width;
        editor_.target_height = height;
        editor_.bind_to(resolved_scene);
        editor_.scan_scenes();
        editor_.scan_textures();

        cached_reset_gen_ =
            config().reset_generation.load(std::memory_order_acquire);

        return true;
    }

    void scene_host::check_pending_reset(render::ray_tracer& rt)
    {
        const std::uint32_t gen =
            config().reset_generation.load(std::memory_order_acquire);
        if (gen != cached_reset_gen_)
        {
            cached_reset_gen_ = gen;
            rt.reset();
        }
    }

    void scene_host::check_pending_editor(render::ray_tracer& rt)
    {
        if (!editor_.pending_load && !editor_.pending_rebuild) return;

        rt.stop();
        container* fresh = editor_.apply_pending();
        if (fresh != nullptr)
        {
            container* old = current_;
            current_ = fresh;
            rt.rebind_scene(current_);
            delete old;

            if (scene_idle_until_loaded_)
            {
                config().pause_render.store(false, std::memory_order_relaxed);
                scene_idle_until_loaded_ = false;
            }
        }
        rt.start();

        cached_reset_gen_ =
            config().reset_generation.load(std::memory_order_acquire);
    }

    void scene_host::release()
    {
        delete current_;
        current_ = nullptr;
    }
} // namespace fox_tracer
