// Multi-actor simulation engine API.

#pragma once

#include "drone_sim.h"
#include "target_sim.h"

#define SIM_MAX_TARGETS 16

typedef struct {
    DroneSim pursuer;
    TargetSim targets[SIM_MAX_TARGETS];
    int num_targets;
    float t;
} SimEngine;

void sim_engine_init(SimEngine* engine, Params params, State pursuer_initial);
void sim_engine_reset(SimEngine* engine, State pursuer_initial);
void sim_engine_clear_targets(SimEngine* engine);
int sim_engine_set_targets(SimEngine* engine, const TargetSim* targets, int num_targets);
int sim_engine_add_target(SimEngine* engine, TargetSim target);
void sim_engine_step_motor_dt(SimEngine* engine, float actions[4], float dt, int substeps);
void sim_engine_step_motor_speeds_dt(SimEngine* engine, float cmd_rpms[4], float dt, int substeps);
State sim_engine_get_pursuer_state(const SimEngine* engine);
int sim_engine_get_num_targets(const SimEngine* engine);
TargetState sim_engine_get_target_state(const SimEngine* engine, int target_index);
