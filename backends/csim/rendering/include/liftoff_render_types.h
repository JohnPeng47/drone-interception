#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum LiftoffRenderBackendKind {
    LIFTOFF_RENDER_BACKEND_NONE = 0,
    LIFTOFF_RENDER_BACKEND_UNITY = 1,
    LIFTOFF_RENDER_BACKEND_SOFTWARE = 2,
} LiftoffRenderBackendKind;

typedef enum LiftoffRenderPlatformKind {
    LIFTOFF_RENDER_PLATFORM_AUTO = 0,
    LIFTOFF_RENDER_PLATFORM_WINDOWS = 1,
    LIFTOFF_RENDER_PLATFORM_LINUX = 2,
} LiftoffRenderPlatformKind;

typedef struct LiftoffRenderConfig {
    LiftoffRenderBackendKind backend;
    LiftoffRenderPlatformKind platform;
    uint32_t timeout_ms;
    uint32_t flags;
    char scene_id[256];
} LiftoffRenderConfig;

typedef struct LiftoffRenderVec3 {
    double x;
    double y;
    double z;
} LiftoffRenderVec3;

typedef struct LiftoffRenderQuatXyzw {
    double x;
    double y;
    double z;
    double w;
} LiftoffRenderQuatXyzw;

typedef struct LiftoffRenderDroneState {
    double t;
    uint64_t sequence_id;
    LiftoffRenderVec3 position_w;
    LiftoffRenderVec3 velocity_w;
    LiftoffRenderQuatXyzw quat_xyzw;
    LiftoffRenderVec3 body_rates_b;
} LiftoffRenderDroneState;

typedef struct LiftoffRenderCameraState {
    uint32_t camera_id;
    LiftoffRenderVec3 position_b;
    double body_to_camera[9];
    uint32_t width_px;
    uint32_t height_px;
    double fx_px;
    double fy_px;
    double cx_px;
    double cy_px;
    double hfov_rad;
    double vfov_rad;
} LiftoffRenderCameraState;

typedef struct LiftoffRenderTargetState {
    uint32_t target_id;
    LiftoffRenderVec3 position_w;
    LiftoffRenderVec3 velocity_w;
    double radius_m;
} LiftoffRenderTargetState;

typedef struct LiftoffRenderFrameRequest {
    const LiftoffRenderDroneState* drone;
    const LiftoffRenderCameraState* camera;
    const LiftoffRenderTargetState* targets;
    uint32_t target_count;
} LiftoffRenderFrameRequest;

typedef struct LiftoffRenderFrame {
    uint64_t sequence_id;
    uint32_t width_px;
    uint32_t height_px;
    uint32_t channels;
    uint32_t stride_bytes;
    const uint8_t* pixels;
    size_t pixel_bytes;
} LiftoffRenderFrame;

#ifdef __cplusplus
}
#endif
