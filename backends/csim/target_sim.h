// Target actor simulation API.

#pragma once

#include "sim_types.h"

#define SIM_MAX_WAYPOINTS 64

typedef enum {
    TARGET_CONTROLLER_LINEAR = 0,
} TargetControllerKind;

typedef enum {
    TARGET_BEHAVIOR_WAYPOINTS = 0,
} TargetBehaviorKind;

typedef struct {
    Vec3 pos;
    Vec3 vel;
} TargetState;

typedef struct {
    Vec3 pos;
    Vec3 vel;
} TargetReference;

typedef struct {
    Vec3 accel;
} TargetCommand;

typedef struct {
    TargetControllerKind kind;
    float kp;
    float kv;
    float max_accel;
} TargetControllerConfig;

typedef struct {
    TargetBehaviorKind kind;
    int num_waypoints;
    Vec3 waypoints[SIM_MAX_WAYPOINTS];
    float duration;
    int loop;
} TargetBehaviorConfig;

typedef struct {
    int id;
    float radius;
    TargetState state;
    TargetBehaviorConfig behavior;
    TargetControllerConfig controller;
} TargetSim;

void target_sim_init(TargetSim* target, int id, float radius, TargetState initial,
                     TargetBehaviorConfig behavior, TargetControllerConfig controller);
void target_sim_reset(TargetSim* target, TargetState initial);
void target_sim_step(TargetSim* target, float t, float dt);
TargetState target_sim_get_state(const TargetSim* target);
TargetReference target_sim_reference(const TargetSim* target, float t);
TargetCommand target_sim_compute_command(const TargetSim* target, TargetReference ref);
