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
#include "render/photon_map.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>

// TODO: kd-tree backend should beat the spatial hash
// TODO: actual hash function for grid_key
// TODO: shrink-to-fit so memory doesnt balloon between passes

namespace fox_tracer::render
{
    void photon_map::init(const float _cell_size) noexcept
    {
        cell_size = _cell_size;
        grid_.clear();
        count_ = 0;
        for (auto& v : pending_) v.clear();
    }

    void photon_map::resize_workers(int num_workers) noexcept
    {
        if (num_workers < 1) num_workers = 1;
        pending_.assign(static_cast<std::size_t>(num_workers),
                        std::vector<photon>{});

        //~ 4096 felt right after profiling smaller and we realloc on every photon burst
        for (auto& v : pending_)
            v.reserve(4096);
    }

    photon_map::grid_key photon_map::pack(int ix, int iy, int iz) noexcept
    {
        const std::uint64_t ux = static_cast<std::uint32_t>(ix) & 0x1FFFFFu;
        const std::uint64_t uy = static_cast<std::uint32_t>(iy) & 0x1FFFFFu;
        const std::uint64_t uz = static_cast<std::uint32_t>(iz) & 0x1FFFFFu;
        return ux | (uy << 21) | (uz << 42);
    }

    photon_map::grid_key photon_map::key_for(const vec3& p) const noexcept
    {
        // const int ix = static_cast<int>(std::floor(p.x / cell_size)); //~ 3 divs per call hella wasteful
        // const int iy = static_cast<int>(std::floor(p.y / cell_size));
        // const int iz = static_cast<int>(std::floor(p.z / cell_size));

        const float inv = 1.0f / cell_size;
        const int ix = static_cast<int>(std::floor(p.x * inv));
        const int iy = static_cast<int>(std::floor(p.y * inv));
        const int iz = static_cast<int>(std::floor(p.z * inv));
        return pack(ix, iy, iz);
    }

    void photon_map::add(const photon& p, int worker_id)
    {
        if (worker_id < 0
            || static_cast<std::size_t>(worker_id) >= pending_.size())
        {
            return;
        }
        pending_[static_cast<std::size_t>(worker_id)].push_back(p);
    }

    void photon_map::merge_pending()
    {
        std::size_t total = 0;
        for (const auto& v : pending_) total += v.size();
        if (total == 0) return;

        // for (auto& v : pending_) //~ no reserve every grid_[k].push_back potentially reallocs
        // {
        //     for (const photon& p : v)
        //     {
        //         const grid_key k = key_for(p.position);
        //         grid_[k].push_back(p);
        //     }
        //     v.clear();
        // }

        // grid_.reserve(grid_.size() + total); //~ way too aggressive total is photon count not cell count
        // for (auto& v : pending_)
        // {
        //     for (const photon& p : v)
        //     {
        //         const grid_key k = key_for(p.position);
        //         grid_[k].push_back(p);
        //     }
        //     v.clear();
        // }

        grid_.reserve(grid_.size() + total / 4);

        for (auto& v : pending_)
        {
            for (const photon& p : v)
            {
                const grid_key k = key_for(p.position);
                grid_[k].push_back(p);
            }
            v.clear();
        }
        count_ += total;
    }

    void photon_map::clear() noexcept
    {
        grid_.clear();
        count_ = 0;
        for (auto& v : pending_) v.clear();
    }

    std::size_t photon_map::size() const noexcept
    {
        return count_;
    }

    void photon_map::knn(const vec3& x, const int k, const float r_max,
                         std::vector<const photon*>& out,
                         float& r_out) const
    {
        out.clear();
        r_out = 0.0f;
        if (k <= 0 || r_max <= 0.0f || grid_.empty()) return;

        const float r2_max = r_max * r_max;

        using qentry = std::pair<float, const photon*>;
        std::priority_queue<qentry> heap;

        const float inv  = 1.0f / cell_size;
        const int   cx_i = static_cast<int>(std::floor(x.x * inv));
        const int   cy_i = static_cast<int>(std::floor(x.y * inv));
        const int   cz_i = static_cast<int>(std::floor(x.z * inv));

        // const int rc = static_cast<int>(std::ceil(r_max / cell_size));
        const int rc = std::max(1, static_cast<int>(std::ceil(r_max / cell_size)));

        // for (int dz = -rc; dz <= rc; ++dz) //~ scanning ALL neighbour cells
        // for (int dy = -rc; dy <= rc; ++dy)
        // for (int dx = -rc; dx <= rc; ++dx)
        // {
        //     const grid_key key = pack(cx_i + dx, cy_i + dy, cz_i + dz);
        //     auto it = grid_.find(key);
        //     if (it == grid_.end()) continue;
        //
        //     for (const photon& p : it->second)
        //     {
        //         const vec3 d = p.position - x;
        //         const float d2 = d.length_squared();
        //         if (d2 > r2_max) continue;
        //
        //         if (static_cast<int>(heap.size()) < k)
        //         {
        //             heap.emplace(d2, &p);
        //         }
        //         else if (d2 < heap.top().first)
        //         {
        //             heap.pop();
        //             heap.emplace(d2, &p);
        //         }
        //     }
        // }

        //~ shrink the search radius
        float r2_cur = r2_max;
        const float cell_sz_sq = cell_size * cell_size;

        for (int dz = -rc; dz <= rc; ++dz)
        for (int dy = -rc; dy <= rc; ++dy)
        for (int dx = -rc; dx <= rc; ++dx)
        {
            const int cell_dx = std::max(0, std::abs(dx) - 1);
            const int cell_dy = std::max(0, std::abs(dy) - 1);
            const int cell_dz = std::max(0, std::abs(dz) - 1);
            const float cell_min_d2 = static_cast<float>(
                cell_dx * cell_dx + cell_dy * cell_dy + cell_dz * cell_dz) * cell_sz_sq;

            if (cell_min_d2 > r2_cur)
                continue;

            const grid_key key = pack(cx_i + dx, cy_i + dy, cz_i + dz);
            auto it = grid_.find(key);

            if (it == grid_.end())
                continue;

            for (const photon& p : it->second)
            {
                const vec3 d = p.position - x;
                const float d2 = d.length_squared();
                if (d2 > r2_cur) continue;

                if (static_cast<int>(heap.size()) < k)
                {
                    heap.emplace(d2, &p);
                    //~ once we hit k top is now a real upper bound tighten r2 cur
                    if (static_cast<int>(heap.size()) == k)
                    {
                        r2_cur = heap.top().first;
                    }
                }
                else if (d2 < heap.top().first)
                {
                    heap.pop();
                    heap.emplace(d2, &p);
                    r2_cur = heap.top().first;
                }
            }
        }

        if (heap.empty()) return;

        const float r2_found = heap.top().first;
        r_out = std::sqrt(r2_found);

        out.reserve(heap.size());
        while (!heap.empty())
        {
            out.push_back(heap.top().second);
            heap.pop();
        }
    }
} // namespace fox_tracer
