// Shared drone simulation core API.
//
// This header exposes the physics/state step independently from the vectorized
// RL task wrapper in drone.h. The existing RL environment can keep using
// DroneEnv/c_step, while Drake/Python adapters can target this smaller API.

#pragma once

#include "sim_types.h"

typedef struct {
    State state;
    Params params;
} DroneSim;

void drone_sim_init(DroneSim* sim, Params params, State initial);
void drone_sim_reset(DroneSim* sim, State initial);
void drone_sim_step_motor(DroneSim* sim, float actions[4]);
void drone_sim_step_motor_dt(DroneSim* sim, float actions[4], float dt, int substeps);
void drone_sim_step_motor_speeds_dt(DroneSim* sim, float cmd_rpms[4], float dt, int substeps);
State drone_sim_get_state(DroneSim* sim);
