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
#include "ui/ui_settings.h"
#include "ui/ui_context.h"

#include "config.h"
#include "render/renderer.h"

#include <imgui.h>
#include <algorithm>
#include <cstdio>

#include "sampler/sampling.h"

namespace fox_tracer::ui
{
    settings_panel::settings_panel(bool* visible) noexcept
        : visible_(visible)
    {}

    void settings_panel::draw(ui_context& ctx)
    {
        if (visible_ != nullptr && !*visible_) return;
        if (!config().show_ui.load(std::memory_order_relaxed)) return;

        ImGui::SetNextWindowPos (ImVec2(10.0f,  30.0f),  ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360.0f, 660.0f), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Settings", visible_))
        {
            ImGui::End();
            return;
        }

        //~ render stats
        const int spp    = ctx.rt->get_spp();
        const int target = std::max(1, config().target_spp.load(
                                       std::memory_order_relaxed));
        const int tiles  = ctx.rt->get_tile_count();

        ImGui::Text("FPS:       %.1f", ctx.fps_ema);
        ImGui::Text("SPP:       %d / %d", spp, target);
        ImGui::Text("SPP / sec: %.2f", ctx.spp_per_sec);
        ImGui::Text("Tiles:     %d", tiles);

        if (ctx.eta_valid)
        {
            const float eta_sec = std::max(0.0f, ctx.eta_sec_ema);
            const int   h = static_cast<int>(eta_sec / 3600.0f);
            const int   m = static_cast<int>((eta_sec - h * 3600.0f) / 60.0f);
            const int   s = static_cast<int>(eta_sec - h * 3600.0f - m * 60.0f);
            if (eta_sec <= 0.0f) ImGui::Text("ETA:        done");
            else                 ImGui::Text("ETA:        %02d:%02d:%02d", h, m, s);
        }
        else
        {
            ImGui::Text("ETA:        --:--:--");
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Sampler", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const char* sampler_names[] =
                {
                "MT random", "Independent",
                "Stratified", "Halton", "Sobol"
            };
            int sk = config().sampler_kind.load();
            if (ImGui::Combo("Sampler kind", &sk, sampler_names,
                             IM_ARRAYSIZE(sampler_names)))
                config().set_sampler_kind(sk);

            const char* presets[] =
            {
                "Quick (Independent)",
                "Balanced (Halton, scrambled, 256 dims)",
                "Best (Sobol, scrambled, 1024 dims)"
            };
            static int preset_idx = -1;
            if (ImGui::Combo("Default preset", &preset_idx, presets,
                             IM_ARRAYSIZE(presets)) && preset_idx >= 0)
            {
                switch (preset_idx)
                {
                case 0:
                    config().set_sampler_kind(static_cast<int>(
                        sampling::sampler_kind::independent));
                    break;
                case 1:
                    config().set_sampler_kind(static_cast<int>(
                        sampling::sampler_kind::halton));
                    config().set_sampler_scrambling(true);
                    config().set_sampler_max_dims(256);
                    break;
                case 2:
                    config().set_sampler_kind(static_cast<int>(
                        sampling::sampler_kind::sobol));
                    config().set_sampler_scrambling(true);
                    config().set_sampler_max_dims(1024);
                    break;
                }
            }

            const auto kind = static_cast<sampling::sampler_kind>(sk);
            ImGui::BeginDisabled(kind != sampling::sampler_kind::stratified);
            int spa = config().sampler_samples_axis.load();
            if (ImGui::SliderInt("Samples per axis", &spa, 1, 16))
                config().set_sampler_samples_axis(spa);
            ImGui::EndDisabled();

            const bool ld = (kind == sampling::sampler_kind::halton ||
                             kind == sampling::sampler_kind::sobol);
            ImGui::BeginDisabled(!ld);
            bool scr = config().sampler_scrambling.load();
            if (ImGui::Checkbox("Scrambling", &scr))
                config().set_sampler_scrambling(scr);
            int md = config().sampler_max_dims.load();
            if (ImGui::SliderInt("Max dimensions", &md, 16, 1024))
                config().set_sampler_max_dims(md);
            ImGui::EndDisabled();
        }

        if (ImGui::CollapsingHeader("Feature Flags", ImGuiTreeNodeFlags_DefaultOpen))
        {
            bool b = config().use_bvh.load();
            if (ImGui::Checkbox("BVH traversal", &b))
                config().set_use_bvh(b);

            const int tile_now = std::max(1, config().tile_size.load());
            int tile_exp = 1;
            while ((1 << tile_exp) < tile_now && tile_exp < 8) ++tile_exp;
            const int displayed_tile = 1 << tile_exp;

            char tile_label[32];
            std::snprintf(tile_label, sizeof(tile_label),
                          "%d px", displayed_tile);
            if (ImGui::SliderInt("Tile size", &tile_exp, 1, 8, tile_label))
            {
                config().set_tile_size(1 << std::clamp(tile_exp, 1, 8));
            }

            const char* filters[] =
            {
                "Mitchell-Netravali", "Gaussian", "Box",
                "Triangle", "Lanczos-Sinc"
            };
            int fk = config().pixel_filter.load();
            if (ImGui::Combo("Pixel filter", &fk, filters, IM_ARRAYSIZE(filters)))
                config().set_pixel_filter(fk);

            const char* presets[] =
            {
                "Sharp (Mitchell B=0, C=0.5, r=2)",
                "Balanced (Mitchell B=1/3, C=1/3, r=2)",
                "Soft (Gaussian r=1.5, alpha=2)",
                "Fast (Box r=0.5)"
            };
            int preset = -1;
            if (ImGui::Combo("Filter preset", &preset,
                presets, IM_ARRAYSIZE(presets))
                && preset >= 0)
            {
                switch (preset)
                {
                case 0:
                    config().set_mitchell_b(0.0f);
                    config().set_mitchell_c(0.5f);
                    config().set_mitchell_radius_x(2.0f);
                    config().set_mitchell_radius_y(2.0f);
                    config().set_pixel_filter(static_cast<int>(pixel_filter_kind::mitchell));
                    break;
                case 1:
                    config().set_mitchell_b(1.0f / 3.0f);
                    config().set_mitchell_c(1.0f / 3.0f);
                    config().set_mitchell_radius_x(2.0f);
                    config().set_mitchell_radius_y(2.0f);
                    config().set_pixel_filter(static_cast<int>(pixel_filter_kind::mitchell));
                    break;
                case 2:
                    config().set_gaussian_radius_x(1.5f);
                    config().set_gaussian_radius_y(1.5f);
                    config().set_gaussian_alpha(2.0f);
                    config().set_pixel_filter(static_cast<int>(pixel_filter_kind::gaussian));
                    break;
                case 3:
                    config().set_box_radius_x(0.5f);
                    config().set_box_radius_y(0.5f);
                    config().set_pixel_filter(static_cast<int>(pixel_filter_kind::box));
                    break;
                default: break;
                }
            }

            const auto kind = static_cast<pixel_filter_kind>(fk);
            if (kind == pixel_filter_kind::mitchell)
            {
                float mb = config().mitchell_b.load();
                float mc = config().mitchell_c.load();
                if (ImGui::SliderFloat("Mitchell B", &mb, 0.0f, 1.0f, "%.3f"))
                    config().set_mitchell_b(mb);
                if (ImGui::SliderFloat("Mitchell C", &mc, 0.0f, 1.0f, "%.3f"))
                    config().set_mitchell_c(mc);

                float r[2] = { config().mitchell_radius_x.load(),
                               config().mitchell_radius_y.load() };
                if (ImGui::DragFloat2("Mitchell radius (px)", r,
                                      0.05f, 0.25f, 4.0f, "%.2f"))
                {
                    config().set_mitchell_radius_x(std::clamp(r[0], 0.25f, 4.0f));
                    config().set_mitchell_radius_y(std::clamp(r[1], 0.25f, 4.0f));
                }
                ImGui::TextDisabled("(B=1/3, C=1/3, r=2) is Mitchell-Netravali default");
            }
            else if (kind == pixel_filter_kind::gaussian)
            {
                float ga = config().gaussian_alpha.load();
                if (ImGui::SliderFloat("Gaussian alpha", &ga, 0.1f, 8.0f, "%.2f"))
                    config().set_gaussian_alpha(ga);

                float r[2] = { config().gaussian_radius_x.load(),
                               config().gaussian_radius_y.load() };
                if (ImGui::DragFloat2("Gaussian radius (px)", r,
                                      0.05f, 0.25f, 4.0f, "%.2f"))
                {
                    config().set_gaussian_radius_x(std::clamp(r[0], 0.25f, 4.0f));
                    config().set_gaussian_radius_y(std::clamp(r[1], 0.25f, 4.0f));
                }
                ImGui::TextDisabled("baked default");
            }
            else if (kind == pixel_filter_kind::box)
            {
                float r[2] = { config().box_radius_x.load(),
                               config().box_radius_y.load() };
                if (ImGui::DragFloat2("Box radius (px)", r,
                                      0.05f, 0.25f, 4.0f, "%.2f"))
                {
                    config().set_box_radius_x(std::clamp(r[0], 0.25f, 4.0f));
                    config().set_box_radius_y(std::clamp(r[1], 0.25f, 4.0f));
                }
                ImGui::TextDisabled("r=0.5 fast path");
            }
            else if (kind == pixel_filter_kind::triangle)
            {
                float r[2] = { config().triangle_radius_x.load(),
                               config().triangle_radius_y.load() };
                if (ImGui::DragFloat2("Triangle radius (px)", r,
                                      0.05f, 0.25f, 4.0f, "%.2f"))
                {
                    config().set_triangle_radius_x(std::clamp(r[0], 0.25f, 4.0f));
                    config().set_triangle_radius_y(std::clamp(r[1], 0.25f, 4.0f));
                }
            }
            else if (kind == pixel_filter_kind::lanczos_sinc)
            {
                float r[2] = { config().lanczos_radius_x.load(),
                               config().lanczos_radius_y.load() };
                if (ImGui::DragFloat2("Lanczos radius (px)", r,
                                      0.05f, 0.25f, 4.0f, "%.2f"))
                {
                    config().set_lanczos_radius_x(std::clamp(r[0], 0.25f, 4.0f));
                    config().set_lanczos_radius_y(std::clamp(r[1], 0.25f, 4.0f));
                }
                float tau = config().lanczos_tau.load();
                if (ImGui::SliderFloat("Lanczos tau", &tau, 1.0f, 4.0f, "%.2f"))
                    config().set_lanczos_tau(tau);
            }

            bool fis = config().use_filter_importance_sampling.load();
            if (ImGui::Checkbox("Use filter importance sampling", &fis))
                config().set_use_filter_importance_sampling(fis);
            ImGui::TextDisabled("Splats one weighted sample per radiance estimate");
        }

        if (ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int   md    = config().max_depth.load();
            int   rrd   = config().rr_depth.load();
            int   tech  = config().sampling_tech.load();
            int   hemi  = config().hemisphere_mode.load();
            int   lpick = config().light_pick_mode.load();
            bool  rr    = config().use_rr.load();

            if (ImGui::SliderInt("Max depth", &md,  1, 64))  config().set_max_depth(md);
            if (ImGui::SliderInt("RR depth",  &rrd, 0, 32))  config().set_rr_depth(rrd);

            const char* sampling_techs[] = { "BSDF only", "NEE", "MIS" };
            if (ImGui::Combo("Sampling", &tech, sampling_techs,
                             static_cast<int>(IM_ARRAYSIZE(sampling_techs))))
                config().set_sampling_tech(tech);

            const char* hemi_modes[] = { "Cosine-weighted", "Uniform" };
            if (ImGui::Combo("Hemisphere sampling", &hemi, hemi_modes,
                             static_cast<int>(IM_ARRAYSIZE(hemi_modes))))
                config().set_hemisphere_mode(hemi);

            const char* lpick_modes[] = { "Uniform", "Importance (power)" };
            if (ImGui::Combo("Light picking", &lpick, lpick_modes,
                             static_cast<int>(IM_ARRAYSIZE(lpick_modes))))
                config().set_light_pick_mode(lpick);

            if (ImGui::Checkbox ("Russian roulette", &rr))
                config().set_use_rr(rr);

            float fd = config().pt_firefly_max_direct  .load();
            float fi = config().pt_firefly_max_indirect.load();
            if (ImGui::SliderFloat("Firefly clamp direct", &fd, 0.0f,
                                   100.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
                config().set_pt_firefly_max_direct(fd);
            if (ImGui::SliderFloat("Firefly clamp indirect", &fi, 0.0f,
                                   100.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
                config().set_pt_firefly_max_indirect(fi);

            const char* modes[] = { "Path trace", "Direct only",
                                    "Albedo", "Normals",
                                    "Instant Radiosity", "Photon Mapping"
            };
            int mode = config().render_mode.load();
            if (ImGui::Combo("Render mode", &mode, modes,
                             static_cast<int>(IM_ARRAYSIZE(modes))))
                config().set_render_mode(mode);
        }

        if (ImGui::CollapsingHeader("Photon Mapping"))
        {
            int pg = config().pm_p_global.load();
            if (ImGui::SliderInt("Photons - global", &pg, 1'000, 2'000'000,
                                 "%d", ImGuiSliderFlags_Logarithmic))
                config().set_pm_p_global(pg);

            int pc = config().pm_p_caustic.load();
            if (ImGui::SliderInt("Photons - caustic", &pc, 500, 1'000'000,
                                 "%d", ImGuiSliderFlags_Logarithmic))
                config().set_pm_p_caustic(pc);

            int kg = config().pm_k_global.load();
            if (ImGui::SliderInt("k - global", &kg, 10, 1000))
                config().set_pm_k_global(kg);

            int kc = config().pm_k_caustic.load();
            if (ImGui::SliderInt("k - caustic", &kc, 10, 1000))
                config().set_pm_k_caustic(kc);

            float rg = config().pm_r_max_global.load();
            if (ImGui::SliderFloat("r_max - global", &rg, 0.001f, 2.0f,
                                   "%.4f", ImGuiSliderFlags_Logarithmic))
                config().set_pm_r_max_global(rg);

            float rc = config().pm_r_max_caustic.load();
            if (ImGui::SliderFloat("r_max - caustic", &rc, 0.001f, 0.5f,
                                   "%.4f", ImGuiSliderFlags_Logarithmic))
                config().set_pm_r_max_caustic(rc);

            bool fg = config().pm_use_final_gather.load();
            if (ImGui::Checkbox("Final gather (global only)", &fg))
                config().set_pm_use_final_gather(fg);

            int fgr = config().pm_final_gather_rays.load();
            if (ImGui::SliderInt("Final gather rays", &fgr, 4, 256))
                config().set_pm_final_gather_rays(fgr);

            if (ImGui::Button("shoot photons again"))
                config().request_reset();
        }

        if (ImGui::CollapsingHeader("Instant Radiosity"))
        {
            int nv = config().ir_num_vpls.load();
            if (ImGui::SliderInt("VPLs per pass", &nv, 32, 4096))
                config().set_ir_num_vpls(nv);

            float md = config().ir_min_dist_sq.load();
            if (ImGui::SliderFloat("Min dist^2 (clamp B)", &md,
                                   1.0e-5f, 1.0f, "%.5f",
                                   ImGuiSliderFlags_Logarithmic))
                config().set_ir_min_dist_sq(md);

            float mc = config().ir_max_contrib.load();
            if (ImGui::SliderFloat("Max contrib (clamp C)", &mc,
                                   0.1f, 100.0f, "%.2f",
                                   ImGuiSliderFlags_Logarithmic))
                config().set_ir_max_contrib(mc);
        }

        if (ImGui::CollapsingHeader("Pinhole Camera(Not Complete)"))
        {
            float fov = config().fov.load();
            if (ImGui::SliderFloat("FOV (vertical)", &fov,
                                   10.0f, 120.0f, "%.1f deg"))
                config().set_fov(fov);

            ImGui::Separator();
            ImGui::TextDisabled("Aperture depth-of-field");

            float lens = config().lens_radius.load();
            if (ImGui::SliderFloat("Aperture lens radius", &lens,
                                   0.0f, 0.5f, "%.4f"))
                config().set_lens_radius(lens);

            const float fdist_now = config().focal_distance.load();
            if (lens > 1.0e-5f && fdist_now > 0.0f)
            {
                const float fnumber = fdist_now / (2.0f * lens);
                ImGui::Text("Approximate f-number: f/%.2f", fnumber);
            }

            float fdist = fdist_now;
            if (ImGui::SliderFloat("Focus distance", &fdist,
                                   0.1f, 100.0f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic))
                config().set_focal_distance(fdist);

            ImGui::Separator();
            float ms = config().move_speed.load();
            if (ImGui::SliderFloat("Move speed", &ms, 0.001f, 100.0f))
                config().move_speed.store(ms);

            float msens = config().mouse_sensitivity.load();
            if (ImGui::SliderFloat("Mouse sensitivity", &msens,
                                   0.00005f, 0.05f, "%.5f",
                                   ImGuiSliderFlags_Logarithmic))
                config().mouse_sensitivity.store(msens);
        }

        //~ TODO: I gotta add animation later
        if (ImGui::CollapsingHeader("Asset Import"))
        {
            bool nrm = config().normalize_obj.load();
            if (ImGui::Checkbox("normalize OBJ to unit cube", &nrm))
                config().normalize_obj.store(nrm);

            float nmax = config().normalize_obj_max.load();
            if (ImGui::SliderFloat("Normalize max extent", &nmax,
                                   0.01f, 100.0f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic))
                config().normalize_obj_max.store(nmax);
        }

        if (ImGui::CollapsingHeader("Tonemap", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::TextDisabled("rengard curve");

            float exposure = config().exposure.load();
            if (ImGui::SliderFloat("Exposure", &exposure, 0.0f, 10.0f))
                config().exposure.store(exposure);

            float gamma = config().gamma.load();
            if (ImGui::SliderFloat("Gamma", &gamma, 0.5f, 4.0f))
                config().gamma.store(gamma);

            float contrast = config().contrast.load();
            if (ImGui::SliderFloat("Contrast", &contrast, 0.1f, 3.0f))
                config().contrast.store(contrast);

            float saturation = config().saturation.load();
            if (ImGui::SliderFloat("Saturation", &saturation, 0.0f, 3.0f))
                config().saturation.store(saturation);
        }

        if (ImGui::CollapsingHeader("Background"))
        {
            bool ovr = config().override_background.load();
            if (ImGui::Checkbox("Override background", &ovr))
                config().set_override_background(ovr);

            float rgb[3] =
                {
                config().bg_r.load(),
                config().bg_g.load(),
                config().bg_b.load()
            };
            if (ImGui::ColorEdit3("Colour", rgb))
            {
                config().set_bg_r(rgb[0]);
                config().set_bg_g(rgb[1]);
                config().set_bg_b(rgb[2]);
            }
        }

        if (ImGui::CollapsingHeader("Dispatcher"))
        {
            int spc = config().samples_per_call.load(std::memory_order_relaxed);
            if (ImGui::SliderInt("Samples per pass", &spc, 1, 4096, "%d",
                                 ImGuiSliderFlags_Logarithmic
                                 | ImGuiSliderFlags_AlwaysClamp))
            {
                config().samples_per_call.store(std::clamp(spc, 1, 4096),
                                                std::memory_order_relaxed);
            }

            int tgt = config().target_spp.load(std::memory_order_relaxed);
            if (ImGui::SliderInt("Target SPP", &tgt, 1, 65536, "%d",
                                 ImGuiSliderFlags_Logarithmic
                                 | ImGuiSliderFlags_AlwaysClamp))
            {
                config().target_spp.store(std::clamp(tgt, 1, 65536),
                                          std::memory_order_relaxed);
            }
            ImGui::TextDisabled("Stop at this many samples per pixel");

            bool pause = config().pause_render.load(std::memory_order_relaxed);
            if (ImGui::Checkbox("Pause render", &pause))
                config().pause_render.store(pause, std::memory_order_relaxed);
        }

        ImGui::End();
    }
} // namespace fox_tracer::ui
