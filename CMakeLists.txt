cmake_minimum_required(VERSION 3.24)

# project props
project(raytracer_with_ai VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# default to release build
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif ()

# get Dependencies
include(FetchContent)

# Imgui
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.5
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)

target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(imgui PUBLIC d3d11 dxgi d3dcompiler)
# imgui end

# stb
FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)

# compile stb once
set(STB_IMPL_FILE ${CMAKE_BINARY_DIR}/stb_impl.c)
file(CONFIGURE OUTPUT ${STB_IMPL_FILE} CONTENT
        "#define STB_IMAGE_IMPLEMENTATION
        #include \"stb_image.h\"
        #define STB_IMAGE_WRITE_IMPLEMENTATION
        #include \"stb_image_write.h\"
        "
)

add_library(stb STATIC ${STB_IMPL_FILE})
target_include_directories(stb PUBLIC ${stb_SOURCE_DIR})
# stb end

# odin
set(OIDN_VERSION 2.3.3)
FetchContent_Declare(
        oidn
        URL https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.x64.windows.zip
)
FetchContent_MakeAvailable(oidn)
add_library(oidn_ext INTERFACE)
target_include_directories(oidn_ext INTERFACE ${oidn_SOURCE_DIR}/include)
target_link_libraries(oidn_ext INTERFACE
        ${oidn_SOURCE_DIR}/lib/OpenImageDenoise.lib
)
target_compile_definitions(oidn_ext INTERFACE FOX_HAS_OIDN=1)
file(GLOB OIDN_RUNTIME_DLLS
        CONFIGURE_DEPENDS
        ${oidn_SOURCE_DIR}/bin/*.dll
)
# odin end

# source
add_executable(tracer_engine src/main.cpp
        src/framework/core.cpp
        src/framework/imaging.cpp
        src/sampler/sampling.cpp
        src/sampler/independent_sampler.cpp
        src/sampler/halton_sampler.cpp
        src/sampler/stratified_sampler.cpp
        src/sampler/sobol_sampler.cpp
        src/framework/materials.cpp
        src/framework/geometry.cpp
        src/framework/lights.cpp
        src/render/instant_radiosity.cpp
        src/utils/helper.cpp
        src/utils/logger.cpp
        src/utils/paths.cpp
        src/utils/assets_loader.cpp
        include/scene/scene_loader.h
        src/scene/scene_loader.cpp
        src/scene/scene.cpp
        src/scene/scene_editor.cpp
        src/utils/frame_stats.cpp
        src/utils/input_controller.cpp
        src/utils/effect_runner.cpp
        src/utils/effect_pipeline.cpp
        src/utils/python_env.cpp
        src/scene/scene_host.cpp
        src/ui/image_gallery.cpp
        src/ui/ai_panel.cpp
        src/ui/ui_denoise.cpp)

target_include_directories(tracer_engine PRIVATE include/)
target_compile_definitions(tracer_engine PRIVATE
        _UNICODE
        UNICODE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
)
target_link_libraries(tracer_engine PRIVATE
        imgui
        stb
        oidn_ext
        d3d11
        d3dcompiler
        WindowsCodecs
        Xinput
)

# source end

# post
foreach(OIDN_DLL ${OIDN_RUNTIME_DLLS})
    add_custom_command(TARGET tracer_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OIDN_DLL}
            $<TARGET_FILE_DIR:tracer_engine>
            VERBATIM
    )
endforeach()

# boost speed
if(MSVC)
    target_compile_options(tracer_engine PRIVATE
            $<$<CONFIG:Release>:/O2 /Oi /Ot /GL /fp:fast /arch:AVX2 /Zc:__cplusplus>
            $<$<CONFIG:RelWithDebInfo>:/O2 /Oi /Ot /GL /fp:fast /arch:AVX2 /Zc:__cplusplus>
    )
    target_link_options(tracer_engine PRIVATE
            $<$<CONFIG:Release>:/LTCG>
            $<$<CONFIG:RelWithDebInfo>:/LTCG>
    )
endif()

# copy assets
file(GLOB_RECURSE ASSET_FILES
        CONFIGURE_DEPENDS
        RELATIVE ${CMAKE_SOURCE_DIR}/assets
        ${CMAKE_SOURCE_DIR}/assets/*
)

foreach(ASSET ${ASSET_FILES})
    add_custom_command(TARGET tracer_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/assets/${ASSET}
            $<TARGET_FILE_DIR:tracer_engine>/assets/${ASSET}
            VERBATIM
    )
endforeach()
# post end
