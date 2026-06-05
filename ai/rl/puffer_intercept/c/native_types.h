#pragma once

#include <stdint.h>

#include "sim_engine.h"

typedef struct {
    int64_t seed;
    State pursuer_initial;
    int num_targets;
    TargetSim targets[SIM_MAX_TARGETS];
    int num_cameras;
    CameraSim cameras[SIM_MAX_CAMERAS];
    PursuerParams pursuer_params;
    float rpm_min;
    float k_w;
    float dt;
    int substeps;
    float duration_s;
    float intercept_radius_m;
    float max_thrust_n;
    float max_rate_rps;
    Vec3 bounds_w;
    int has_bounds;
} NativeScenario;
