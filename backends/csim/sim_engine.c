#include "sim_engine.h"

#include <math.h>

#include "sim_math.h"

static InterceptMetrics empty_metrics(void) {
    return (InterceptMetrics){
        .distance_m = 0.0f,
        .min_distance_m = INFINITY,
        .intercepted = 0,
        .intercept_time_s = -1.0f,
        .target_index = -1,
    };
}

static void sim_engine_update_metrics(SimEngine* engine) {
    if (engine->num_targets <= 0) {
        engine->metrics.distance_m = 0.0f;
        return;
    }

    Vec3 pursuer_pos = engine->pursuer.state.pos;
    float best_distance = INFINITY;
    int best_index = -1;
    for (int i = 0; i < engine->num_targets; i++) {
        Vec3 target_pos = engine->targets[i].state.pos;
        Vec3 delta = sub3(target_pos, pursuer_pos);
        float distance = norm3(delta);
        if (distance < best_distance) {
            best_distance = distance;
            best_index = i;
        }
    }

    engine->metrics.distance_m = best_distance;
    engine->metrics.target_index = best_index;
    if (best_distance < engine->metrics.min_distance_m) {
        engine->metrics.min_distance_m = best_distance;
    }
    if (!engine->metrics.intercepted &&
            engine->intercept_radius_m > 0.0f &&
            best_distance <= engine->intercept_radius_m) {
        engine->metrics.intercepted = 1;
        engine->metrics.intercept_time_s = engine->t;
    }
}

void sim_engine_init(SimEngine* engine, Params params, State pursuer_initial) {
    pursuer_sim_init(&engine->pursuer, params, pursuer_initial);
    engine->num_targets = 0;
    engine->num_cameras = 0;
    engine->t = 0.0f;
    engine->intercept_radius_m = 0.0f;
    engine->metrics = empty_metrics();
}

void sim_engine_reset(SimEngine* engine, State pursuer_initial) {
    pursuer_sim_reset(&engine->pursuer, pursuer_initial);
    engine->t = 0.0f;
    engine->metrics = empty_metrics();
    sim_engine_update_metrics(engine);
}

void sim_engine_set_intercept_radius(SimEngine* engine, float intercept_radius_m) {
    engine->intercept_radius_m = intercept_radius_m > 0.0f ? intercept_radius_m : 0.0f;
    sim_engine_update_metrics(engine);
}

void sim_engine_clear_targets(SimEngine* engine) {
    engine->num_targets = 0;
}

void sim_engine_clear_cameras(SimEngine* engine) {
    engine->num_cameras = 0;
}

int sim_engine_set_targets(SimEngine* engine, const TargetSim* targets, int num_targets) {
    if (num_targets < 0) num_targets = 0;
    if (num_targets > SIM_MAX_TARGETS) num_targets = SIM_MAX_TARGETS;

    for (int i = 0; i < num_targets; i++) {
        engine->targets[i] = targets[i];
    }
    engine->num_targets = num_targets;
    sim_engine_update_metrics(engine);
    return num_targets;
}

int sim_engine_add_target(SimEngine* engine, TargetSim target) {
    if (engine->num_targets >= SIM_MAX_TARGETS) return -1;

    int idx = engine->num_targets;
    engine->targets[idx] = target;
    engine->num_targets++;
    sim_engine_update_metrics(engine);
    return idx;
}

int sim_engine_set_cameras(SimEngine* engine, const CameraSim* cameras, int num_cameras) {
    if (num_cameras < 0) num_cameras = 0;
    if (num_cameras > SIM_MAX_CAMERAS) num_cameras = SIM_MAX_CAMERAS;

    for (int i = 0; i < num_cameras; i++) {
        engine->cameras[i] = cameras[i];
        camera_sim_reset(&engine->cameras[i]);
    }
    engine->num_cameras = num_cameras;
    return num_cameras;
}

int sim_engine_add_camera(SimEngine* engine, CameraSim camera) {
    if (engine->num_cameras >= SIM_MAX_CAMERAS) return -1;

    int idx = engine->num_cameras;
    engine->cameras[idx] = camera;
    camera_sim_reset(&engine->cameras[idx]);
    engine->num_cameras++;
    return idx;
}

int sim_engine_collect_camera_outputs(SimEngine* engine, CameraOutput* outputs, int max_outputs) {
    if (outputs == NULL || max_outputs <= 0) return 0;
    int count = 0;

    for (int c = 0; c < engine->num_cameras && count < max_outputs; c++) {
        CameraSim* camera = &engine->cameras[c];
        if (!camera_sim_capture_due(camera, engine->t)) continue;

        CameraOutput output = {0};
        output.has_frame = 0;
        output.frame_width_px = camera->intrinsics.width_px;
        output.frame_height_px = camera->intrinsics.height_px;
        output.frame_channels = 3;
        output.frame_rgb = NULL;

        if (engine->num_targets <= 0) {
            output.observation.camera_id = camera->id;
            output.observation.target_index = -1;
            output.observation.captured = 1;
            output.observation.detected = 0;
            output.observation.t_capture = engine->t;
            outputs[count++] = output;
            continue;
        }

        output.observation = camera_sim_observe_target(
            camera,
            &engine->pursuer.state,
            &engine->targets[0],
            0,
            engine->t
        );
        outputs[count++] = output;
    }

    return count;
}

void sim_engine_step_motor_dt(SimEngine* engine, float actions[4], float dt, int substeps) {
    pursuer_sim_step_motor_dt(&engine->pursuer, actions, dt, substeps);
    for (int i = 0; i < engine->num_targets; i++) {
        target_sim_step(&engine->targets[i], engine->t, dt);
    }
    engine->t += dt;
    sim_engine_update_metrics(engine);
}

void sim_engine_step_motor_speeds_dt(SimEngine* engine, float cmd_rpms[4], float dt, int substeps) {
    pursuer_sim_step_motor_speeds_dt(&engine->pursuer, cmd_rpms, dt, substeps);
    for (int i = 0; i < engine->num_targets; i++) {
        target_sim_step(&engine->targets[i], engine->t, dt);
    }
    engine->t += dt;
    sim_engine_update_metrics(engine);
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

InterceptMetrics sim_engine_get_metrics(const SimEngine* engine) {
    return engine->metrics;
}
