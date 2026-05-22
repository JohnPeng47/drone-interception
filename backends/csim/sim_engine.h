// Multi-actor simulation engine API.

#pragma once

#include "pursuer_sim.h"
#include "camera_sim.h"
#include "target_sim.h"

#define SIM_MAX_TARGETS 16
#define SIM_MAX_CAMERAS 8
#define SIM_MAX_CAMERA_OUTPUTS 8

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
} SimEngine;

void sim_engine_init(SimEngine* engine, Params params, State pursuer_initial);
void sim_engine_reset(SimEngine* engine, State pursuer_initial);
void sim_engine_set_intercept_radius(SimEngine* engine, float intercept_radius_m);
void sim_engine_clear_targets(SimEngine* engine);
int sim_engine_set_targets(SimEngine* engine, const TargetSim* targets, int num_targets);
int sim_engine_add_target(SimEngine* engine, TargetSim target);
void sim_engine_clear_cameras(SimEngine* engine);
int sim_engine_set_cameras(SimEngine* engine, const CameraSim* cameras, int num_cameras);
int sim_engine_add_camera(SimEngine* engine, CameraSim camera);
int sim_engine_collect_camera_outputs(SimEngine* engine, CameraOutput* outputs, int max_outputs);
void sim_engine_step_motor_dt(SimEngine* engine, float actions[4], float dt, int substeps);
void sim_engine_step_motor_speeds_dt(SimEngine* engine, float cmd_rpms[4], float dt, int substeps);
State sim_engine_get_pursuer_state(const SimEngine* engine);
int sim_engine_get_num_targets(const SimEngine* engine);
TargetState sim_engine_get_target_state(const SimEngine* engine, int target_index);
InterceptMetrics sim_engine_get_metrics(const SimEngine* engine);
