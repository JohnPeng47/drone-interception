cmake_minimum_required(VERSION 3.16)
project(liftoff_render_native LANGUAGES CXX)

add_library(liftoff_render_native SHARED
    src/render_engine.cpp
)

target_compile_features(liftoff_render_native PRIVATE cxx_std_17)
target_include_directories(liftoff_render_native
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/../include
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/platform
        ${CMAKE_CURRENT_LIST_DIR}/src
)

if(WIN32)
    target_sources(liftoff_render_native PRIVATE
        platform/win32/render_platform_win32.cpp
    )
    target_compile_definitions(liftoff_render_native PRIVATE LIFTOFF_RENDER_PLATFORM_WIN32=1)
else()
    target_sources(liftoff_render_native PRIVATE
        platform/linux/render_platform_linux.cpp
    )
    target_compile_definitions(liftoff_render_native PRIVATE LIFTOFF_RENDER_PLATFORM_LINUX=1)
endif()
