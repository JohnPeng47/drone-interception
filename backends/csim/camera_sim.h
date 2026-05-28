// Camera projection/capture API.

#pragma once

#include <stddef.h>

#include "target_sim.h"

typedef struct {
    int width_px;
    int height_px;
    float fx_px;
    float fy_px;
    float cx_px;
    float cy_px;
    float hfov_rad;
    float vfov_rad;
} CameraIntrinsics;

typedef struct {
    int id;
    int parent_actor;
    Vec3 position_b;
    Mat3 body_to_camera;
    CameraIntrinsics intrinsics;
    float capture_rate_hz;
    float next_capture_t;
} CameraSim;

typedef struct {
    int camera_id;
    int target_index;
    int captured;
    int detected;
    float t_capture;
    Vec3 target_pos_c;
    float range_m;
    float uv_norm[2];
    float uv_px[2];
    float apparent_radius_px;
} CameraObservation;

typedef struct {
    CameraObservation observation;
    int has_frame;
    int frame_width_px;
    int frame_height_px;
    int frame_channels;
    int frame_stride_bytes;
    size_t frame_byte_count;
    int render_status;
    const unsigned char* frame_rgb;
} CameraOutput;

void camera_sim_reset(CameraSim* camera);
int camera_sim_capture_due(CameraSim* camera, float t);
CameraObservation camera_sim_observe_target(
    const CameraSim* camera,
    const State* pursuer,
    const TargetSim* target,
    int target_index,
    float t
);
