#include "sim_engine.h"

void sim_engine_init(SimEngine* engine, Params params, State pursuer_initial) {
    drone_sim_init(&engine->pursuer, params, pursuer_initial);
    engine->num_targets = 0;
    engine->t = 0.0f;
}

void sim_engine_reset(SimEngine* engine, State pursuer_initial) {
    drone_sim_reset(&engine->pursuer, pursuer_initial);
    engine->t = 0.0f;
}

void sim_engine_clear_targets(SimEngine* engine) {
    engine->num_targets = 0;
}

int sim_engine_set_targets(SimEngine* engine, const TargetSim* targets, int num_targets) {
    if (num_targets < 0) num_targets = 0;
    if (num_targets > SIM_MAX_TARGETS) num_targets = SIM_MAX_TARGETS;

    for (int i = 0; i < num_targets; i++) {
        engine->targets[i] = targets[i];
    }
    engine->num_targets = num_targets;
    return num_targets;
}

int sim_engine_add_target(SimEngine* engine, TargetSim target) {
    if (engine->num_targets >= SIM_MAX_TARGETS) return -1;

    int idx = engine->num_targets;
    engine->targets[idx] = target;
    engine->num_targets++;
    return idx;
}

void sim_engine_step_motor_dt(SimEngine* engine, float actions[4], float dt, int substeps) {
    drone_sim_step_motor_dt(&engine->pursuer, actions, dt, substeps);
    for (int i = 0; i < engine->num_targets; i++) {
        target_sim_step(&engine->targets[i], engine->t, dt);
    }
    engine->t += dt;
}

void sim_engine_step_motor_speeds_dt(SimEngine* engine, float cmd_rpms[4], float dt, int substeps) {
    drone_sim_step_motor_speeds_dt(&engine->pursuer, cmd_rpms, dt, substeps);
    for (int i = 0; i < engine->num_targets; i++) {
        target_sim_step(&engine->targets[i], engine->t, dt);
    }
    engine->t += dt;
}

State sim_engine_get_pursuer_state(const SimEngine* engine) {
    return engine->pursuer.state;
}

int sim_engine_get_num_targets(const SimEngine* engine) {
    return engine->num_targets;
}

TargetState sim_engine_get_target_state(const SimEngine* engine, int target_index) {
    if (target_index < 0 || target_index >= engine->num_targets) {
        return (TargetState){0};
    }
    return target_sim_get_state(&engine->targets[target_index]);
}
