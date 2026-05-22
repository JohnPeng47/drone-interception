// Low-level quadrotor simulation API.
//
// PursuerSim owns only the physical vehicle state and vehicle parameters. Higher
// level world orchestration lives in SimEngine.

#pragma once

#include "sim_types.h"

typedef struct {
    State state;
    Params params;
} PursuerSim;

void pursuer_sim_init(PursuerSim* sim, Params params, State initial);
void pursuer_sim_reset(PursuerSim* sim, State initial);
void pursuer_sim_step_motor(PursuerSim* sim, float actions[4]);
void pursuer_sim_step_motor_dt(PursuerSim* sim, float actions[4], float dt, int substeps);
void pursuer_sim_step_motor_speeds_dt(PursuerSim* sim, float cmd_rpms[4], float dt, int substeps);
State pursuer_sim_get_state(PursuerSim* sim);
