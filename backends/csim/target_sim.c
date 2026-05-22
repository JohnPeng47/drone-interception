#include "target_sim.h"

#include "sim_math.h"

static TargetReference hold_reference(const TargetSim* target) {
    return (TargetReference){target->state.pos, (Vec3){0.0f, 0.0f, 0.0f}};
}

void target_sim_init(TargetSim* target, int id, float radius, TargetState initial,
                     TargetBehaviorConfig behavior, TargetControllerConfig controller) {
    target->id = id;
    target->radius = radius;
    target->state = initial;
    target->behavior = behavior;
    target->controller = controller;

    if (target->behavior.num_waypoints < 0) target->behavior.num_waypoints = 0;
    if (target->behavior.num_waypoints > SIM_MAX_WAYPOINTS) {
        target->behavior.num_waypoints = SIM_MAX_WAYPOINTS;
    }
}

void target_sim_reset(TargetSim* target, TargetState initial) {
    target->state = initial;
}

void target_sim_step(TargetSim* target, float t, float dt) {
    TargetReference ref = target_sim_reference(target, t);
    TargetCommand cmd = target_sim_compute_command(target, ref);
    target->state.vel = add3(target->state.vel, scalmul3(cmd.accel, dt));
    target->state.pos = add3(target->state.pos, scalmul3(target->state.vel, dt));
}

TargetState target_sim_get_state(const TargetSim* target) {
    return target->state;
}

TargetReference target_sim_reference(const TargetSim* target, float t) {
    const TargetBehaviorConfig* behavior = &target->behavior;
    if (behavior->num_waypoints <= 0) {
        return hold_reference(target);
    }

    int n = behavior->num_waypoints;
    if (n == 1 || behavior->duration <= 0.0f) {
        return (TargetReference){behavior->waypoints[0], (Vec3){0.0f, 0.0f, 0.0f}};
    }

    float local_t = t;
    if (behavior->loop) {
        local_t = fmodf(local_t, behavior->duration);
        if (local_t < 0.0f) local_t += behavior->duration;
    } else if (local_t >= behavior->duration) {
        return (TargetReference){behavior->waypoints[n - 1], (Vec3){0.0f, 0.0f, 0.0f}};
    } else if (local_t < 0.0f) {
        local_t = 0.0f;
    }

    int segments = n - 1;
    float segment_dt = behavior->duration / (float)segments;
    int segment = (int)floorf(local_t / segment_dt);
    if (segment < 0) segment = 0;
    if (segment >= segments) segment = segments - 1;

    Vec3 start = behavior->waypoints[segment];
    Vec3 end = behavior->waypoints[segment + 1];
    Vec3 delta = sub3(end, start);
    float segment_t = local_t - (float)segment * segment_dt;
    float u = clampf(segment_t / segment_dt, 0.0f, 1.0f);

    return (TargetReference){
        add3(start, scalmul3(delta, u)),
        scalmul3(delta, 1.0f / segment_dt),
    };
}

TargetCommand target_sim_compute_command(const TargetSim* target, TargetReference ref) {
    const TargetControllerConfig* controller = &target->controller;
    Vec3 pos_error = sub3(ref.pos, target->state.pos);
    Vec3 vel_error = sub3(ref.vel, target->state.vel);
    Vec3 accel = add3(scalmul3(pos_error, controller->kp),
                      scalmul3(vel_error, controller->kv));

    float accel_norm = norm3(accel);
    if (controller->max_accel > 0.0f && accel_norm > controller->max_accel) {
        accel = scalmul3(accel, controller->max_accel / accel_norm);
    }

    return (TargetCommand){accel};
}
