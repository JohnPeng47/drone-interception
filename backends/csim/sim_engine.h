// Multi-actor simulation engine API.

#pragma once

#include "camera_sim.h"
#include "liftoff_render_api.h"
#include "sim_types.h"
#include "target_sim.h"

#define SIM_MAX_TARGETS 16
#define SIM_MAX_CAMERAS 8
#define SIM_MAX_CAMERA_OUTPUTS 8
#define SIM_SNAPSHOT_PURSUER_SIZE 17
#define SIM_SNAPSHOT_TARGET_SIZE 6
#define SIM_SNAPSHOT_METRICS_SIZE 5
#define SIM_SNAPSHOT_CAMERA_SIZE 3

typedef struct {
    float distance_m;
    float min_distance_m;
    int intercepted;
    float intercept_time_s;
    int target_index;
} InterceptMetrics;

typedef struct {
    PursuerSim pursuer;
    TargetSim targets[SIM_MAX_TARGETS];
    CameraSim cameras[SIM_MAX_CAMERAS];
    int num_targets;
    int num_cameras;
    float t;
    float intercept_radius_m;
    InterceptMetrics metrics;
    int render_enabled;
    int render_camera_id;
    int render_fail_on_error;
    LiftoffRenderConfig render_config;
    LiftoffRenderEngine* render_engine;
    unsigned long long render_sequence_id;
    unsigned char* render_frame_buffers[SIM_MAX_CAMERA_OUTPUTS];
    size_t render_frame_buffer_bytes[SIM_MAX_CAMERA_OUTPUTS];
} SimEngine;

typedef struct {
    float t;
    State pursuer_state;
    int num_targets;
    TargetState target_states[SIM_MAX_TARGETS];
    int target_ids[SIM_MAX_TARGETS];
    float target_radii_m[SIM_MAX_TARGETS];
    float intercept_radius_m;
    InterceptMetrics metrics;
    int num_camera_outputs;
    CameraOutput camera_outputs[SIM_MAX_CAMERA_OUTPUTS];
} SimSnapshot;

typedef struct {
    int num_engines;
    float* pursuer_state;
    float* first_target_state;
    float* metrics;
    float* first_camera_observation;
    float* max_rate_rps;
    float* max_rpm;
} SimSnapshots;

void sim_engine_init(SimEngine* engine, PursuerParams params, State pursuer_initial);
void sim_engine_reset(SimEngine* engine, State pursuer_initial);
void sim_engine_set_intercept_radius(SimEngine* engine, float intercept_radius_m);
void sim_engine_clear_targets(SimEngine* engine);
int sim_engine_set_targets(SimEngine* engine, const TargetSim* targets, int num_targets);
int sim_engine_add_target(SimEngine* engine, TargetSim target);
void sim_engine_clear_cameras(SimEngine* engine);
int sim_engine_set_cameras(SimEngine* engine, const CameraSim* cameras, int num_cameras);
int sim_engine_add_camera(SimEngine* engine, CameraSim camera);
LiftoffRenderStatus sim_engine_configure_rendering(
    SimEngine* engine,
    int enabled,
    int camera_id,
    int fail_on_error,
    const LiftoffRenderConfig* config
);
void sim_engine_close_rendering(SimEngine* engine);
int sim_engine_collect_camera_outputs(SimEngine* engine, CameraOutput* outputs, int max_outputs);
void sim_engine_step_motor_dt(SimEngine* engine, float actions[4], float dt, int substeps);
void sim_engine_step_motor_speeds_dt(SimEngine* engine, float cmd_rpms[4], float dt, int substeps);
State sim_engine_get_pursuer_state(const SimEngine* engine);
int sim_engine_get_num_targets(const SimEngine* engine);
TargetState sim_engine_get_target_state(const SimEngine* engine, int target_index);
InterceptMetrics sim_engine_get_metrics(const SimEngine* engine);
void sim_engine_get_snapshot(SimEngine* engine, SimSnapshot* snapshot);

void sim_engine_batch_step_motor_speeds_dt(
    SimEngine* engines,
    const float* cmd_rpms,
    int num_engines,
    float dt,
    int substeps
);
void sim_engine_batch_get_snapshots(
    SimEngine* engines,
    int num_engines,
    SimSnapshots* snapshots
);
