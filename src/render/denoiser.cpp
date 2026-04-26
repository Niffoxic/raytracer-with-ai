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
#include "render/denoiser.h"
#include "utils/logger.h"
#include <OpenImageDenoise/oidn.hpp>

namespace fox_tracer::render
{
    namespace
    {
        inline oidn::DeviceRef* as_device(void* p) noexcept
        {
            return static_cast<oidn::DeviceRef*>(p);
        }
    }

    denoiser::denoiser() noexcept = default;

    denoiser::~denoiser()
    {
        if (device_ != nullptr)
        {
            delete as_device(device_);
            device_ = nullptr;
        }
    }

    bool denoiser::ensure_ready()
    {
        if (device_ != nullptr) return true;

        auto* dev = new oidn::DeviceRef(oidn::newDevice(oidn::DeviceType::CPU));
        dev->commit();

        const char* err_msg = nullptr;
        if (dev->getError(err_msg) != oidn::Error::None)
        {
            LOG_ERROR("denoiser") << "OIDN device init failed: "
                                  << (err_msg ? err_msg : "unknown");
            delete dev;
            return false;
        }
        device_ = dev;
        LOG_INFO("denoiser") << "OIDN CPU device ready";
        return true;
    }

    bool denoiser::denoise(const color* color_in,
                           const color* albedo_in,
                           const color* normal_in,
                           color*       color_out,
                           int          width,
                           int          height,
                           bool         hdr)
    {
        if (!ensure_ready())      return false;
        if (color_in  == nullptr) return false;
        if (color_out == nullptr) return false;
        if (width <= 0 || height <= 0) return false;

        oidn::DeviceRef& dev = *as_device(device_);

        oidn::FilterRef filter = dev.newFilter("RT");
        const std::size_t pixel_stride = sizeof(color);
        const std::size_t row_stride   = pixel_stride * static_cast<std::size_t>(width);

        filter.setImage("color",  const_cast<color*>(color_in),
                        oidn::Format::Float3, width, height,
                        0, pixel_stride, row_stride);

        if (use_albedo && albedo_in != nullptr)
        {
            filter.setImage("albedo", const_cast<color*>(albedo_in),
                            oidn::Format::Float3, width, height,
                            0, pixel_stride, row_stride);
        }
        if (use_normal && normal_in != nullptr)
        {
            filter.setImage("normal", const_cast<color*>(normal_in),
                            oidn::Format::Float3, width, height,
                            0, pixel_stride, row_stride);
        }

        filter.setImage("output", color_out,
                        oidn::Format::Float3, width, height,
                        0, pixel_stride, row_stride);

        filter.set("hdr", hdr);
        filter.commit();

        filter.execute();

        const char* err_msg = nullptr;
        if (dev.getError(err_msg) != oidn::Error::None)
        {
            LOG_ERROR("denoiser") << "OIDN filter execute failed: "
                                  << (err_msg ? err_msg : "unknown");
            return false;
        }
        return true;
    }
} // namespace fox_tracer
