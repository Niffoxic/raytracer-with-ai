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
#ifndef RAYTRACER_WITH_AI_ASSETS_LOADER_H
#define RAYTRACER_WITH_AI_ASSETS_LOADER_H

#include "gem_loader.h"

#include <memory>
#include <string>
#include <vector>

namespace fox_tracer::loader
{
    using gem_vec3          = GEMLoader::GEMVec3;
    using gem_static_vertex = GEMLoader::GEMStaticVertex;
    using gem_mesh          = GEMLoader::GEMMesh;
    using gem_property      = GEMLoader::GEMProperty;
    using gem_material      = GEMLoader::GEMMaterial;
    using gem_instance      = GEMLoader::GEMInstance;
    using gem_scene         = GEMLoader::GEMScene;
    using gem_matrix        = GEMLoader::GEMMatrix;

    class interface_model
    {
    public:
        virtual ~interface_model() = default;

        virtual bool load(const std::string& filename,
                          std::vector<gem_mesh>& meshes) = 0;
    };

    struct obj_face
    {
        int v[3];
        int t[3];
        int n[3];
    };

    class obj_model
    {
    public:
        bool load(const std::string& filename,
                  std::vector<gem_mesh>& meshes);

    private:
        static void     parse_face_token(const std::string& token,
                                         int pos_size, int tex_size, int nrm_size,
                                         int& pi, int& ti, int& ni);

        static gem_vec3 to_gem_vec(float x, float y, float z);
        static gem_vec3 sub_vec   (const gem_vec3& a, const gem_vec3& b);
        static gem_vec3 cross_vec (const gem_vec3& a, const gem_vec3& b);

        static gem_vec3 normalize_vec(const gem_vec3& a);
    };

    class obj_model_adapter: public interface_model
    {
    public:
        bool load(const std::string& filename,
                  std::vector<gem_mesh>& meshes) override;

    private:
        obj_model impl;
    };

    class gem_model_adapter : public interface_model
    {
    public:
        bool load(const std::string& filename,
                  std::vector<gem_mesh>& meshes) override;

    private:
        GEMLoader::GEMModelLoader adaptee;
    };

    class glb_model_adapter: public interface_model
    {
    public:
        bool load(const std::string& filename,
                  std::vector<gem_mesh>& meshes) override;
    };

    std::unique_ptr<interface_model> make_model_loader(const std::string& filename);
} // namespace fox_tracer
#endif //RAYTRACER_WITH_AI_ASSETS_LOADER_H
