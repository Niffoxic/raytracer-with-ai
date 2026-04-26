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

#include "utils/input_controller.h"

#include "config.h"
#include "scene/scene_loader.h"
#include "framework/base.h"

#include <windows.h>
#include <algorithm>

namespace fox_tracer::utility
{
    void input_controller::process(GamesEngineeringBase::Window* window)
    {
        if (window == nullptr) return;

        const bool rmb_down =
            window->mouseButtonPressed(GamesEngineeringBase::MouseRight);

        if (rmb_down && !window->imguiWantsMouse())
        {
            const int cx = static_cast<int>(window->getClientWidth())  / 2;
            const int cy = static_cast<int>(window->getClientHeight()) / 2;
            if (!mouse_look_active_)
            {
                mouse_look_active_ = true;
                window->setCursorVisible(false);
                window->warpCursorTo(cx, cy);
            }
            else
            {
                const int mx = window->getMouseInWindowX();
                const int my = window->getMouseInWindowY();
                const int dx = mx - cx;
                const int dy = my - cy;
                if (dx != 0 || dy != 0)
                {
                    view_camera().look(dx, dy);
                    config().request_reset();
                    window->warpCursorTo(cx, cy);
                }
            }
        }
        else if (mouse_look_active_)
        {
            mouse_look_active_ = false;
            window->setCursorVisible(true);
        }

        if (window->imguiWantsTextInput() && !mouse_look_active_) return;

        view_camera().movespeed =
            std::max(0.0f, config().move_speed.load(std::memory_order_relaxed));

        const float sprint = window->keyPressed(VK_SHIFT) ? 3.0f : 1.0f;

        bool moved = false;
        if (window->keyPressed('W'))
        {
            view_camera().forward(sprint);
            moved = true;
        }
        if (window->keyPressed('S'))
        {
            view_camera().back(sprint);
            moved = true;
        }
        if (window->keyPressed('A'))
        {
            view_camera().strafe_left(sprint);
            moved = true;
        }
        if (window->keyPressed('D'))
        {
            view_camera().strafe_right(sprint);
            moved = true;
        }
        if (window->keyPressed(VK_SPACE))
        {
            view_camera().fly_up  (sprint);
            moved = true;
        }
        if (window->keyPressed(VK_CONTROL))
        {
            view_camera().fly_down(sprint);
            moved = true;
        }
        if (window->keyPressed('Q'))
        {
            view_camera().fly_down(sprint);
            moved = true;
        }
        if (window->keyPressed('E'))
        {
            view_camera().fly_up(sprint);
            moved = true;
        }
        if (moved) config().request_reset();
    }
} // namespace fox_tracer
