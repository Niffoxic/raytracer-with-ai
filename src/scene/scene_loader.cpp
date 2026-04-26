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
#include "scene/scene_loader.h"
#include "framework/materials.h"

#include "utils/logger.h"
#include "utils/assets_loader.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "utils/helper.h"
#include "utils/paths.h"

void fox_tracer::rt_camera::bind_to_scene(
    camera *c, const vec3 &_from, const vec3 &_to, const vec3 &_up)
{
    cam  = c;
    from = _from;
    to   = _to;
    up   = _up.length_squared() > 0.0f ? _up.normalize() : vec3(0.0f, 1.0f, 0.0f);

    const vec3 fwd = (to - from).normalize();

    pitch = std::asin(std::clamp(fwd.y, -1.0f, 1.0f));
    yaw   = std::atan2(fwd.x, -fwd.z);

    initialized = true;
}

void fox_tracer::rt_camera::forward(const float dt_scale)
{
    const vec3 fwd = math::yaw_pitch_forward(yaw, pitch);
    from = from + fwd * (movespeed * dt_scale);
    apply_view();
}

void fox_tracer::rt_camera::back(const float dt_scale)
{
    const vec3 fwd = math::yaw_pitch_forward(yaw, pitch);
    from = from - fwd * (movespeed * dt_scale);
    apply_view();
}

void fox_tracer::rt_camera::strafe_left(const float dt_scale)
{
    const vec3 r =  math::yaw_pitch_right(yaw);
    from = from - r * (movespeed * dt_scale);
    apply_view();
}

void fox_tracer::rt_camera::strafe_right(const float dt_scale)
{
    const vec3 r = math::yaw_pitch_right(yaw);
    from = from + r * (movespeed * dt_scale);
    apply_view();
}

void fox_tracer::rt_camera::fly_up(const float dt_scale)
{
    from = from + up * (movespeed * dt_scale);
    apply_view();
}

void fox_tracer::rt_camera::fly_down(const float dt_scale)
{
    from = from - up * (movespeed * dt_scale);
    apply_view();
}

void fox_tracer::rt_camera::left()
{
    yaw -= rotspeed * (math::pi<float> / 180.0f);
    apply_view();
}

void fox_tracer::rt_camera::right()
{
    yaw += rotspeed * (math::pi<float> / 180.0f);
    apply_view();
}

void fox_tracer::rt_camera::apply_view()
{
    if (cam == nullptr) return;
    to       = from + math::yaw_pitch_forward(yaw, pitch);
    matrix V =  transform::look_at(from, to, up);
    V = V.invert();
    cam->update_view(V);
}

void fox_tracer::rt_camera::look(const int dx_px, const int dy_px)
{
    const float sens = std::max(0.00005f,
        config().mouse_sensitivity.load(std::memory_order_relaxed));

    yaw   += static_cast<float>(dx_px) * sens;
    pitch -= static_cast<float>(dy_px) * sens;

    constexpr float limit = math::pi<float> * 0.5f - 0.01f;

    if (pitch >  limit) pitch =  limit;
    if (pitch < -limit) pitch = -limit;

    apply_view();
}

fox_tracer::rt_camera & fox_tracer::view_camera() noexcept
{
    static rt_camera vc;
    return vc;
}

fox_tracer::bsdf::factory & fox_tracer::bsdf::factory::global() noexcept
{
    static factory inst;
    static const bool _ = (install_defaults(inst), true);
    (void)_;
    return inst;
}

void fox_tracer::bsdf::factory::register_creator(
    const std::string &name, creator_fn fn)
{
    creators_[name] = std::move(fn);
}

fox_tracer::bsdf::base * fox_tracer::bsdf::factory::create(
    const std::string &name, const loader::gem_material &mat,
    const std::string &scene_dir, texture_cache &tex_cache) const
{
    const auto it = creators_.find(name);
    if (it == creators_.end()) return nullptr;
    return it->second(mat, scene_dir, tex_cache);
}

namespace fox_tracer
{
    namespace
    {
        std::string resolve_tex(const loader::gem_material& mat,
                               const std::string& key,
                               const std::string& scene_dir)
        {
            const std::string rel = const_cast<loader::gem_material&>(mat)
                                        .find(key).getValue(std::string{});
            if (rel.empty()) return scene_dir + "/";

            if (rel.size() > 1 && (rel[0] == '/' || rel[0] == '\\'
                                   || (rel.size() > 2 && rel[1] == ':')))
            {
                return rel;
            }

            const std::string textures_root =
                paths::resolve(std::string("assets/textures/") + rel);

            if (paths::exists(textures_root)) return textures_root;

            const std::string scene_local = scene_dir + "/" + rel;
            return scene_local;
        }

        bsdf::base* make_diffuse(const loader::gem_material& mat,
        const std::string& scene_dir, texture_cache& tex_cache)
        {
            return new fox_tracer::bsdf::diffuse(
                tex_cache.get_or_load(resolve_tex(mat,
                    "reflectance", scene_dir)));
        }

        bsdf::base* make_orennayar(const loader::gem_material& mat,
                             const std::string& scene_dir,
                             texture_cache& tex_cache)
        {
            const float alpha = const_cast<loader::gem_material&>(mat).find("alpha").getValue(1.0f);
            return new bsdf::oren_nayar(
                tex_cache.get_or_load(resolve_tex(mat,
                    "reflectance", scene_dir)),
                alpha);
        }

        bsdf::base* make_glass(const loader::gem_material& mat,
                         const std::string& scene_dir,
                         texture_cache& tex_cache)
        {
            const float int_ior = const_cast<loader::gem_material&>(mat).find("intIOR").getValue(1.33f);
            const float ext_ior = const_cast<loader::gem_material&>(mat).find("extIOR").getValue(1.0f);
            return new bsdf::glass(
                tex_cache.get_or_load(resolve_tex(mat,
                    "reflectance", scene_dir)),
                int_ior, ext_ior);
        }

        bsdf::base* make_mirror(const loader::gem_material& mat,
                          const std::string& scene_dir,
                          texture_cache& tex_cache)
        {
            return new bsdf::mirror(
                tex_cache.get_or_load(resolve_tex(mat,
                    "reflectance", scene_dir)));
        }

        bsdf::base* make_plastic(const loader::gem_material& mat,
                           const std::string& scene_dir,
                           texture_cache& tex_cache)
        {
            const float int_ior   = const_cast<loader::gem_material&>(mat).find("intIOR").getValue(1.33f);
            const float ext_ior   = const_cast<loader::gem_material&>(mat).find("extIOR").getValue(1.0f);
            const float roughness = const_cast<loader::gem_material&>(mat).find("roughness").getValue(1.0f);
            return new bsdf::plastic(
                tex_cache.get_or_load(resolve_tex(mat,
                    "reflectance", scene_dir)),
                int_ior, ext_ior, roughness);
        }

        bsdf::base* make_dielectric(const loader::gem_material& mat,
                              const std::string& scene_dir,
                              texture_cache& tex_cache)
        {
            const float int_ior   = const_cast<loader::gem_material&>(mat).find("intIOR").getValue(1.33f);
            const float ext_ior   = const_cast<loader::gem_material&>(mat).find("extIOR").getValue(1.0f);
            const float roughness = const_cast<loader::gem_material&>(mat).find("roughness").getValue(1.0f);
            texture* tex = tex_cache.get_or_load(resolve_tex(mat, "reflectance", scene_dir));

            if (roughness < 0.001f)
            {
                return new bsdf::glass(tex, int_ior, ext_ior);
            }
            return new bsdf::dielectric(tex, int_ior, ext_ior, roughness);
        }

        bsdf::base* make_conductor(const loader::gem_material& mat,
                             const std::string& scene_dir,
                             texture_cache& tex_cache)
        {
            color eta;
            color k;
            const_cast<loader::gem_material&>(mat).find("eta").getValuesAsVector3(eta.red, eta.green, eta.blue);
            const_cast<loader::gem_material&>(mat).find("k").getValuesAsVector3(k.red, k.green, k.blue);
            const float roughness = const_cast<loader::gem_material&>(mat).find("roughness").getValue(1.0f);

            return new bsdf::conductor(
                tex_cache.get_or_load(resolve_tex(mat, "reflectance", scene_dir)),
                eta, k, roughness);
        }

         static void load_instance(const std::string&           scene_name,
                              std::vector<geometry::triangle>&  mesh_triangles,
                              std::vector<bsdf::base*>&         mesh_materials,
                              loader::gem_instance&             instance,
                              int                               instance_idx,
                              texture_cache&                    texture_manager)
        {
            std::vector<loader::gem_mesh> meshes;
            std::string mesh_path;
            bool        load_ok = false;
            {
                mesh_path = scene_name + "/" + instance.meshFilename;
                if (!paths::exists(mesh_path))
                {
                    LOG_ERROR("scene")
                        << "instance #" << instance_idx
                        << " mesh file not found: " << mesh_path;;
                    return;
                }
                auto loader = loader::make_model_loader(mesh_path);
                load_ok = loader->load(mesh_path, meshes);
            }

            const bool  cfg_norm  = config().normalize_obj.load(std::memory_order_relaxed);
            const float cfg_unit  = config().normalize_obj_max.load(std::memory_order_relaxed);
            const int   per_inst  = instance.material.find("normalize").getValue(-1);

            const bool  do_norm   = helper::ends_with_ci(instance.meshFilename, ".obj")
                                    && (per_inst == 1 || (per_inst != 0 && cfg_norm));

            if (do_norm && !meshes.empty())
            {
                float bmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
                float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                for (const auto& m : meshes)
                {
                    for (const auto& v : m.verticesStatic)
                    {
                        bmin[0] = std::min(bmin[0], v.position.x);
                        bmin[1] = std::min(bmin[1], v.position.y);
                        bmin[2] = std::min(bmin[2], v.position.z);
                        bmax[0] = std::max(bmax[0], v.position.x);
                        bmax[1] = std::max(bmax[1], v.position.y);
                        bmax[2] = std::max(bmax[2], v.position.z);
                    }
                }
                const float sx = bmax[0] - bmin[0];
                const float sy = bmax[1] - bmin[1];
                const float sz = bmax[2] - bmin[2];
                const float largest = std::max(sx, std::max(sy, sz));
                if (largest > 0.0f)
                {
                    const float scale = cfg_unit / largest;
                    const float cx = 0.5f * (bmin[0] + bmax[0]);
                    const float cy = 0.5f * (bmin[1] + bmax[1]);
                    const float cz = 0.5f * (bmin[2] + bmax[2]);
                    for (auto& m : meshes)
                    {
                        for (auto& v : m.verticesStatic)
                        {
                            v.position.x = (v.position.x - cx) * scale;
                            v.position.y = (v.position.y - cy) * scale;
                            v.position.z = (v.position.z - cz) * scale;
                        }
                    }
                }
            }

            if (!meshes.empty())
            {
                const auto& src_props = meshes.front().material.properties;
                for (const auto& p : src_props)
                {
                    bool already = false;
                    for (const auto& dp : instance.material.properties)
                        if (dp.name == p.name) { already = true; break; }
                    if (!already) instance.material.properties.push_back(p);
                }
            }

            std::string bsdf_name = instance.material.find("bsdf").getValue(std::string{});
            if (bsdf_name.empty()
                && !instance.material.find("reflectance").getValue(std::string{}).empty())
            {
                bsdf_name = "diffuse";
            }

            if (bsdf_name.empty()
                && instance.material.find("intIOR").getValue(0.0f) > 0.0f)
            {
                bsdf_name = "glass";
            }
            bsdf::base* material = bsdf::factory::global().create(
                bsdf_name, instance.material,
                scene_name, texture_manager);

            if (material == nullptr)
            {
                LOG_WARN("scene") << "unknown bsdf '" << bsdf_name
                                  << "', falling back to diffuse";
                material = bsdf::factory::global().create(
                    "diffuse", instance.material,
                    scene_name, texture_manager);
            }

            if (material != nullptr)
            {
                mesh_materials.push_back(material);
            }

            if (instance.material.find("emission").getValue(std::string{}) != "")
            {
                color emission;
                instance.material.find("emission")
                                 .getValuesAsVector3(
                                     emission.red,
                                     emission.green,
                                     emission.blue);
                if (material != nullptr) material->add_light(emission);
            }

            if (instance.material.find("coatingThickness").getValue(0) > 0)
            {
                bsdf::base* base = material;
                color sigma_a;
                instance.material.find("coatingSigmaA")
                                 .getValuesAsVector3(sigma_a.red, sigma_a.green, sigma_a.blue);

                const float int_ior   = instance.material.find("coatingIntIOR").getValue(1.33f);
                const float ext_ior   = instance.material.find("coatingExtIOR").getValue(1.0f);
                const float thickness = instance.material.find("coatingThickness").getValue(0.0f);

                material = new bsdf::layered(base, sigma_a, thickness, int_ior, ext_ior);
                if (!mesh_materials.empty()) mesh_materials.back() = material;
            }

            if (material == nullptr)
            {
                LOG_ERROR("scene") << "unknown bsdf '" << bsdf_name << "'";
                return;
            }

            const int material_index = static_cast<int>(mesh_materials.size()) - 1;

            float uv_rot_deg = instance.material.find("uv_rot").getValue(0.0f);
            float uv_sc_u = 1.0f, uv_sc_v = 1.0f;

            if (instance.material.find("uv_scale").getValue(std::string{}) != "")
            {
                float tmp = 0.0f;
                instance.material.find("uv_scale").getValuesAsVector3(uv_sc_u, uv_sc_v, tmp);

                if (uv_sc_u == 0.0f) uv_sc_u = 1.0f;
                if (uv_sc_v == 0.0f) uv_sc_v = 1.0f;
            }
            float uv_off_u = 0.0f, uv_off_v = 0.0f;

            if (instance.material.find("uv_offset").getValue(std::string{}) != "")
            {
                float tmp = 0.0f;
                instance.material.find("uv_offset").getValuesAsVector3(uv_off_u, uv_off_v, tmp);
            }
            const float uv_rot_rad = uv_rot_deg * math::pi<float> / 180.0f;
            const float uv_cos     = std::cos(uv_rot_rad);
            const float uv_sin     = std::sin(uv_rot_rad);

            const bool  uv_identity =
                uv_rot_deg == 0.0f && uv_sc_u == 1.0f && uv_sc_v == 1.0f
                && uv_off_u == 0.0f && uv_off_v == 0.0f;

            auto xform_uv = [&](float u, float v, float& ou, float& ov)
            {
                const float du = u - 0.5f;
                const float dv = v - 0.5f;
                const float ru =  uv_cos * du - uv_sin * dv;
                const float rv =  uv_sin * du + uv_cos * dv;
                ou = (ru + 0.5f) * uv_sc_u + uv_off_u;
                ov = (rv + 0.5f) * uv_sc_v + uv_off_v;
            };

            std::vector<vertex>       vertices;
            std::vector<unsigned int> indices;

            matrix transform;
            std::memcpy(transform.m, instance.w.m, 16 * sizeof(float));
            matrix vec_transform = transform.invert();
            vec_transform = vec_transform.transpose();

            for (size_t i = 0; i < meshes.size(); ++i)
            {
                const auto vertex_base = static_cast<unsigned int>(vertices.size());
                for (size_t n = 0; n < meshes[i].verticesStatic.size(); ++n)
                {
                    const auto& sv = meshes[i].verticesStatic[n];
                    vertex v;
                    v.position.x = sv.position.x;
                    v.position.y = sv.position.y;
                    v.position.z = sv.position.z;
                    v.normal.x = sv.normal.x;
                    v.normal.y = sv.normal.y;
                    v.normal.z = sv.normal.z;

                    v.position = transform.mul_point(v.position);
                    v.normal  = vec_transform.mul_vec(v.normal).normalize();

                    if (uv_identity)
                    {
                        v.u = sv.u;
                        v.v = sv.v;
                    }
                    else
                    {
                        xform_uv(sv.u, sv.v, v.u, v.v);
                    }
                    vertices.push_back(v);
                }
                for (size_t n = 0; n < meshes[i].indices.size(); ++n)
                {
                    indices.push_back(vertex_base + meshes[i].indices[n]);
                }
            }

            const std::size_t tri_before = mesh_triangles.size();
            std::size_t       degenerate = 0;
            float bmin_x =  FLT_MAX, bmin_y =  FLT_MAX, bmin_z =  FLT_MAX;
            float bmax_x = -FLT_MAX, bmax_y = -FLT_MAX, bmax_z = -FLT_MAX;

            for (size_t i = 0; i + 2 < indices.size(); i += 3)
            {
                geometry::triangle t;
                t.init(vertices[indices[i]],
                       vertices[indices[i + 1]],
                       vertices[indices[i + 2]],
                       static_cast<unsigned int>(material_index));

                if (t.area > 0.0f)
                {
                    mesh_triangles.push_back(t);
                    for (int k = 0; k < 3; ++k)
                    {
                        const auto& p = vertices[indices[i + k]].position;
                        bmin_x = std::min(bmin_x, p.x);
                        bmin_y = std::min(bmin_y, p.y);
                        bmin_z = std::min(bmin_z, p.z);
                        bmax_x = std::max(bmax_x, p.x);
                        bmax_y = std::max(bmax_y, p.y);
                        bmax_z = std::max(bmax_z, p.z);
                    }
                }
                else
                {
                    ++degenerate;
                }
            }
            const std::size_t tri_added = mesh_triangles.size() - tri_before;

            if (tri_added == 0)
            {
                //~ TODO: WHAT M I SUPPosed to do now
            }
        }
    } // namespace
}


void fox_tracer::bsdf::factory::install_defaults(factory &f)
{
    f.register_creator("diffuse",    make_diffuse);
    f.register_creator("orennayar",  make_orennayar);
    f.register_creator("glass",      make_glass);
    f.register_creator("mirror",     make_mirror);
    f.register_creator("plastic",    make_plastic);
    f.register_creator("dielectric", make_dielectric);
    f.register_creator("conductor",  make_conductor);
}

fox_tracer::scene::container * fox_tracer::build_scene_from_gem(
    loader::gem_scene &gemscene,
    const std::string &scene_name,
    int override_w, int override_h)
{
    auto* sc = new scene::container();

    int   width  = gemscene.findProperty("width") .getValue(1920);
    int   height = gemscene.findProperty("height").getValue(1080);
    float fov    = gemscene.findProperty("fov")   .getValue(45.0f);

    if (override_w > 0) width  = override_w;
    if (override_h > 0) height = override_h;

    matrix P = transform::perspective(0.001f, 10000.0f,
                                        static_cast<float>(width)
                                      / static_cast<float>(height),
                                        fov);
    vec3 from, to, up;
    gemscene.findProperty("from").getValuesAsVector3(from.x, from.y, from.z);
    gemscene.findProperty("to")  .getValuesAsVector3(to.x,   to.y,   to.z);
    gemscene.findProperty("up")  .getValuesAsVector3(up.x,   up.y,   up.z);
    matrix V = transform::look_at(from, to, up).invert();

    const int flip = gemscene.findProperty("flipX").getValue(0);
    if (flip == 1)
    {
        P.a[0][0] = -P.a[0][0];
    }

    sc->cam.init(P, width, height);
    sc->cam.flip_x  = (flip == 1);
    sc->cam.fov_deg = fov;
    sc->cam.update_view(V);

    view_camera().bind_to_scene(&sc->cam, from, to, up);

    std::vector<geometry::triangle> mesh_triangles;
    std::vector<bsdf::base*>    mesh_materials;

    sc->textures = std::make_unique<texture_cache>();
    texture_cache& texture_manager = *sc->textures;

    const int num_instances = static_cast<int>(gemscene.instances.size());

    if (num_instances > 1)
    {
        int num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads < 1) num_threads = 1;
        if (num_threads > num_instances) num_threads = num_instances;

        std::vector<std::vector<geometry::triangle>> per_inst_tris(num_instances);
        std::vector<std::vector<bsdf::base*>>    per_inst_mats(num_instances);
        std::atomic<int> next_instance(0);

        auto worker = [&]()
        {
            while (true)
            {
                const int i = next_instance.fetch_add(1);
                if (i >= num_instances) break;
                load_instance(scene_name,
                              per_inst_tris[i], per_inst_mats[i],
                              gemscene.instances[i],
                              i,
                              texture_manager);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) workers.emplace_back(worker);
        for (auto& th : workers) th.join();

        for (int i = 0; i < num_instances; ++i)
        {
            const int material_offset = static_cast<int>(mesh_materials.size());
            for (auto* m : per_inst_mats[i]) mesh_materials.push_back(m);
            for (auto& t : per_inst_tris[i])
            {
                t.material_index = static_cast<unsigned int>(
                    static_cast<int>(t.material_index) + material_offset);
                mesh_triangles.push_back(t);
            }
        }
    }
    else
    {
        for (int i = 0; i < num_instances; ++i)
        {
            load_instance(scene_name, mesh_triangles, mesh_materials,
                          gemscene.instances[i], i, texture_manager);
        }
    }

    // Background.
    lights::base* background;
    const std::string envmap_rel =
        gemscene.findProperty("envmap").getValue(std::string{});
    if (!envmap_rel.empty())
    {
        const std::string textures_root =
            paths::resolve(std::string("assets/textures/") + envmap_rel);
        const std::string chosen = paths::exists(textures_root)
                                    ? textures_root
                                    : (scene_name + "/" + envmap_rel);
        texture* env = texture_manager.get_or_load(chosen);
        background = new lights::environment_map(env);
    }
    else
    {
        background = new lights::background_colour(color(0.0f, 0.0f, 0.0f));
    }

    sc->init(mesh_triangles, mesh_materials, background);


    int n_diffuse = 0, n_mirror = 0, n_conductor = 0, n_glass = 0;
    int n_dielectric = 0, n_oren = 0, n_plastic = 0, n_layered = 0;
    for (bsdf::base* m : mesh_materials)
    {
        if      (dynamic_cast<bsdf::diffuse*>   (m)) ++n_diffuse;
        else if (dynamic_cast<bsdf::mirror*>    (m)) ++n_mirror;
        else if (dynamic_cast<bsdf::conductor*> (m)) ++n_conductor;
        else if (dynamic_cast<bsdf::glass*>     (m)) ++n_glass;
        else if (dynamic_cast<bsdf::dielectric*>(m)) ++n_dielectric;
        else if (dynamic_cast<bsdf::oren_nayar*>(m)) ++n_oren;
        else if (dynamic_cast<bsdf::plastic*>   (m)) ++n_plastic;
        else if (dynamic_cast<bsdf::layered*>   (m)) ++n_layered;
    }
    if (mesh_materials.size() < static_cast<std::size_t>(num_instances))
    {
        LOG_ERROR("scene")
            << (num_instances - static_cast<int>(mesh_materials.size()))
            << " load failure occured";
    }

    if (mesh_triangles.empty())
    {
        sc->bounds.min = vec3(-1.0f, -1.0f, -1.0f);
        sc->bounds.max = vec3( 1.0f,  1.0f,  1.0f);
    }

    view_camera().movespeed =
        (sc->bounds.max - sc->bounds.min).length() * 0.05f;
    if (!(view_camera().movespeed > 0.0f) || view_camera().movespeed > 1e6f)
    {
        view_camera().movespeed = 1.0f;
    }

    sc->build();

    singleton::use<scene_bounds>().scene_centre =
        (sc->bounds.max + sc->bounds.min) * 0.5f;
    singleton::use<scene_bounds>().scene_radius =
        (sc->bounds.max - singleton::use<scene_bounds>().scene_centre).length();

    return sc;
}

fox_tracer::scene::container * fox_tracer::load_scene(
    const std::string &scene_name, int override_w,
    int override_h)
{
    const std::string scene_json = scene_name + "/scene.json";
    if (!paths::exists(scene_json))
    {
        LOG_ERROR("scene") << "scene.json not found: " << scene_json;
    }
    loader::gem_scene gemscene;
    gemscene.load(scene_json);
    scene::container* sc = build_scene_from_gem(gemscene, scene_name,
                                     override_w, override_h);
    if (sc != nullptr)
    {
        LOG_INFO("scene") << "loaded " << sc->triangles.size()
                          << " triangles, " << sc->materials.size()
                          << " materials from " << scene_name;
    }
    return sc;
}
