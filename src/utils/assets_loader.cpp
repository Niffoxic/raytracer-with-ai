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

#include "utils/assets_loader.h"

#include "utils/logger.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

namespace fox_tracer::loader
{
    void obj_model::parse_face_token(
        const std::string& token, const int pos_size,
        const int tex_size, const int nrm_size,
        int& pi, int& ti, int& ni)
    {
        pi = -1; ti = -1; ni = -1;
        int  slot = 0;
        int  sign = 1;
        int  val  = 0;
        bool have = false;

        auto commit = [&](const int which)
        {
            if (!have) return;
            int v = val * sign;
            if (v < 0)
            {
                if      (which == 0) v = pos_size + v + 1;
                else if (which == 1) v = tex_size + v + 1;
                else                 v = nrm_size + v + 1;
            }
            if      (which == 0) pi = v;
            else if (which == 1) ti = v;
            else                 ni = v;
        };

        for (const char c : token)
        {
            if (c == '/')
            {
                commit(slot);
                ++slot;
                sign = 1; val = 0; have = false;
            }
            else if (c == '-')
            {
                sign = -1;
            }
            else if (c >= '0' && c <= '9')
            {
                val = val * 10 + (c - '0');
                have = true;
            }
        }
        commit(slot);
    }

    gem_vec3 obj_model::to_gem_vec(float x, float y, float z)
    {
        gem_vec3 v;
        v.x = x; v.y = y; v.z = z;
        return v;
    }

    gem_vec3 obj_model::sub_vec(const gem_vec3& a, const gem_vec3& b)
    {
        return to_gem_vec(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    gem_vec3 obj_model::cross_vec(const gem_vec3& a, const gem_vec3& b)
    {
        return to_gem_vec(a.y * b.z - a.z * b.y,
                          a.z * b.x - a.x * b.z,
                          a.x * b.y - a.y * b.x);
    }

    gem_vec3 obj_model::normalize_vec(const gem_vec3& a)
    {
        const float l = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        if (l < 1e-12f) return to_gem_vec(0.0f, 1.0f, 0.0f);
        const float inv = 1.0f / l;
        return to_gem_vec(a.x * inv, a.y * inv, a.z * inv);
    }

    bool obj_model::load(const std::string& filename,
                                std::vector<gem_mesh>& meshes)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            LOG_ERROR("obj") << "cannot open: " << filename;
            return false;
        }

        std::vector<gem_vec3>                positions;
        std::vector<gem_vec3>                normals;
        std::vector<std::pair<float, float>> texcoords;

        struct submesh_buf
        {
            std::vector<std::vector<int>> face_vs;
            std::vector<std::vector<int>> face_ts;
            std::vector<std::vector<int>> face_ns;
        };

        std::vector<submesh_buf> submeshes;
        submeshes.emplace_back();
        submesh_buf* current = &submeshes.back();

        auto begin_submesh = [&]()
        {
            if (!current->face_vs.empty())
            {
                submeshes.emplace_back();
                current = &submeshes.back();
            }
        };

        std::string line;
        while (std::getline(file, line))
        {
            if (const size_t hash = line.find('#'); hash != std::string::npos)
                line = line.substr(0, hash);

            if (line.empty()) continue;

            std::istringstream ss(line);
            std::string tag;
            ss >> tag;

            if (tag == "v")
            {
                float x, y, z;
                ss >> x >> y >> z;
                positions.push_back(to_gem_vec(x, y, z));
            }
            else if (tag == "vn")
            {
                float x, y, z;
                ss >> x >> y >> z;
                normals.push_back(to_gem_vec(x, y, z));
            }
            else if (tag == "vt")
            {
                float u = 0.0f;
                float v = 0.0f;
                ss >> u >> v;
                texcoords.emplace_back(u, v);
            }
            else if (tag == "o" || tag == "g" || tag == "usemtl")
            {
                begin_submesh();
            }
            else if (tag == "f")
            {
                std::vector<int> fv, ft, fn;
                std::string tok;
                while (ss >> tok)
                {
                    int pi, ti, ni;
                    parse_face_token(tok,
                                     static_cast<int>(positions.size()),
                                     static_cast<int>(texcoords.size()),
                                     static_cast<int>(normals.size()),
                                     pi, ti, ni);
                    fv.push_back(pi);
                    ft.push_back(ti);
                    fn.push_back(ni);
                }
                if (fv.size() >= 3)
                {
                    current->face_vs.push_back(std::move(fv));
                    current->face_ts.push_back(std::move(ft));
                    current->face_ns.push_back(std::move(fn));
                }
            }
        }
        file.close();

        if (positions.empty())
        {
            LOG_WARN("obj") << "no geometry in: " << filename;
            return false;
        }

        bool emitted_any = false;
        for (auto& sub : submeshes)
        {
            if (sub.face_vs.empty()) continue;

            gem_mesh mesh;
            for (size_t f = 0; f < sub.face_vs.size(); ++f)
            {
                const auto& fv = sub.face_vs[f];
                const auto& ft = sub.face_ts[f];
                const auto& fn = sub.face_ns[f];

                gem_vec3 face_n = to_gem_vec(0.0f, 0.0f, 0.0f);
                if (fv.size() >= 3 && fv[0] >= 1 && fv[1] >= 1 && fv[2] >= 1
                    && fv[0] <= static_cast<int>(positions.size())
                    && fv[1] <= static_cast<int>(positions.size())
                    && fv[2] <= static_cast<int>(positions.size()))
                {
                    const auto& p0 = positions[fv[0] - 1];
                    const auto& p1 = positions[fv[1] - 1];
                    const auto& p2 = positions[fv[2] - 1];
                    face_n = normalize_vec(cross_vec(sub_vec(p1, p0), sub_vec(p2, p0)));
                }

                for (size_t k = 1; k + 1 < fv.size(); ++k)
                {
                    const int tri[3] = { 0, static_cast<int>(k), static_cast<int>(k + 1) };
                    const auto base_index =
                        static_cast<unsigned int>(mesh.verticesStatic.size());

                    for (int c : tri)
                    {
                        gem_static_vertex v;

                        const int pi = fv[c];
                        if (pi < 1 || pi > static_cast<int>(positions.size()))
                        {
                            v.position = to_gem_vec(0.0f, 0.0f, 0.0f);
                        }
                        else
                        {
                            v.position = positions[pi - 1];
                        }

                        const int ni = (c < static_cast<int>(fn.size())) ? fn[c] : -1;
                        if (ni >= 1 && ni <= static_cast<int>(normals.size()))
                        {
                            v.normal = normalize_vec(normals[ni - 1]);
                        }
                        else
                        {
                            v.normal = face_n;
                        }

                        v.tangent = to_gem_vec(0.0f, 0.0f, 0.0f);

                        const int ti = (c < static_cast<int>(ft.size())) ? ft[c] : -1;
                        if (ti >= 1 && ti <= static_cast<int>(texcoords.size()))
                        {
                            v.u = texcoords[ti - 1].first;
                            v.v = 1.0f - texcoords[ti - 1].second;
                        }
                        else
                        {
                            v.u = 0.0f;
                            v.v = 0.0f;
                        }

                        mesh.verticesStatic.push_back(v);
                    }
                    mesh.indices.push_back(base_index + 0);
                    mesh.indices.push_back(base_index + 1);
                    mesh.indices.push_back(base_index + 2);
                }
            }

            if (!mesh.verticesStatic.empty() && !mesh.indices.empty())
            {
                meshes.push_back(std::move(mesh));
                emitted_any = true;
            }
        }

        if (!emitted_any)
        {
            LOG_WARN("obj") << "no geometry in: " << filename;
            return false;
        }
        return true;
    }

    bool obj_model_adapter::load(const std::string& filename,
                                        std::vector<gem_mesh>& meshes)
    {
        return impl.load(filename, meshes);
    }

    bool gem_model_adapter::load(const std::string& filename,
                                        std::vector<gem_mesh>& meshes)
    {
        const size_t before = meshes.size();
        adaptee.load(filename, meshes);
        return meshes.size() > before;
    }

    std::unique_ptr<interface_model> make_model_loader(const std::string& filename)
    {
        const size_t dot = filename.find_last_of('.');
        std::string  ext = (dot == std::string::npos)
                         ? std::string{}
                         : filename.substr(dot + 1);
        std::ranges::transform(ext, ext.begin(),
       [](const unsigned char c)
           {
               return static_cast<char>(std::tolower(c));
           });

        if (ext == "obj")
        {
            return std::make_unique<obj_model_adapter>();
        }
        if (ext == "glb")
        {
            return std::make_unique<glb_model_adapter>();
        }
        return std::make_unique<gem_model_adapter>();
    }
} // namespace fox_tracer
