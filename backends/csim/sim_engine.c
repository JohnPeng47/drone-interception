#include "sim_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static void sim_engine_init_render_fields(SimEngine* engine) {
    engine->render_enabled = 0;
    engine->render_camera_id = -1;
    engine->render_fail_on_error = 0;
    memset(&engine->render_config, 0, sizeof(engine->render_config));
    engine->render_engine = NULL;
    engine->render_sequence_id = 0;
    for (int i = 0; i < SIM_MAX_CAMERA_OUTPUTS; i++) {
        engine->render_frame_buffers[i] = NULL;
        engine->render_frame_buffer_bytes[i] = 0;
    }
}

static void sim_engine_free_render_buffers(SimEngine* engine) {
    for (int i = 0; i < SIM_MAX_CAMERA_OUTPUTS; i++) {
        free(engine->render_frame_buffers[i]);
        engine->render_frame_buffers[i] = NULL;
        engine->render_frame_buffer_bytes[i] = 0;
    }
}

static int sim_engine_render_camera_selected(const SimEngine* engine, const CameraSim* camera) {
    return engine->render_enabled &&
        engine->render_engine != NULL &&
        (engine->render_camera_id < 0 || engine->render_camera_id == camera->id);
}

static LiftoffRenderVec3 render_vec3(Vec3 value) {
    return (LiftoffRenderVec3){
        .x = value.x,
        .y = value.y,
        .z = value.z,
    };
}

static LiftoffRenderDroneState sim_engine_render_drone_state(SimEngine* engine) {
    State state = engine->pursuer.state;
    engine->render_sequence_id++;
    return (LiftoffRenderDroneState){
        .t = engine->t,
        .sequence_id = (uint64_t)engine->render_sequence_id,
        .position_w = render_vec3(state.pos),
        .velocity_w = render_vec3(state.vel),
        .quat_xyzw = {
            .x = state.quat.x,
            .y = state.quat.y,
            .z = state.quat.z,
            .w = state.quat.w,
        },
        .body_rates_b = render_vec3(state.omega),
    };
}

static LiftoffRenderCameraState sim_engine_render_camera_state(const CameraSim* camera) {
    LiftoffRenderCameraState out = {
        .camera_id = (uint32_t)camera->id,
        .position_b = render_vec3(camera->position_b),
        .width_px = (uint32_t)camera->intrinsics.width_px,
        .height_px = (uint32_t)camera->intrinsics.height_px,
        .fx_px = camera->intrinsics.fx_px,
        .fy_px = camera->intrinsics.fy_px,
        .cx_px = camera->intrinsics.cx_px,
        .cy_px = camera->intrinsics.cy_px,
        .hfov_rad = camera->intrinsics.hfov_rad,
        .vfov_rad = camera->intrinsics.vfov_rad,
    };
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            out.body_to_camera[r * 3 + c] = camera->body_to_camera.m[r][c];
        }
    }
    return out;
}

static void sim_engine_render_target_states(
    const SimEngine* engine,
    LiftoffRenderTargetState* targets
) {
    for (int i = 0; i < engine->num_targets; i++) {
        targets[i] = (LiftoffRenderTargetState){
            .target_id = (uint32_t)engine->targets[i].id,
            .position_w = render_vec3(engine->targets[i].state.pos),
            .velocity_w = render_vec3(engine->targets[i].state.vel),
            .radius_m = engine->targets[i].radius,
        };
    }
}

static void sim_engine_render_camera_output(
    SimEngine* engine,
    const CameraSim* camera,
    CameraOutput* output,
    int output_index
) {
    output->render_status = -1;
    if (!sim_engine_render_camera_selected(engine, camera)) {
        return;
    }

    LiftoffRenderDroneState drone = sim_engine_render_drone_state(engine);
    LiftoffRenderCameraState render_camera = sim_engine_render_camera_state(camera);
    LiftoffRenderTargetState targets[SIM_MAX_TARGETS];
    sim_engine_render_target_states(engine, targets);
    LiftoffRenderFrameRequest request = {
        .drone = &drone,
        .camera = &render_camera,
        .targets = targets,
        .target_count = (uint32_t)engine->num_targets,
    };
    LiftoffRenderFrame frame = {0};
    LiftoffRenderStatus status = liftoff_render_frame(engine->render_engine, &request, &frame);
    output->render_status = status;
    if (status != LIFTOFF_RENDER_OK || frame.pixels == NULL || frame.pixel_bytes == 0) {
        liftoff_render_release_frame(engine->render_engine, &frame);
        return;
    }

    if (output_index < 0 || output_index >= SIM_MAX_CAMERA_OUTPUTS) {
        liftoff_render_release_frame(engine->render_engine, &frame);
        return;
    }
    if (engine->render_frame_buffer_bytes[output_index] < frame.pixel_bytes) {
        unsigned char* resized = (unsigned char*)realloc(
            engine->render_frame_buffers[output_index],
            frame.pixel_bytes
        );
        if (resized == NULL) {
            output->render_status = LIFTOFF_RENDER_INTERNAL_ERROR;
            liftoff_render_release_frame(engine->render_engine, &frame);
            return;
        }
        engine->render_frame_buffers[output_index] = resized;
        engine->render_frame_buffer_bytes[output_index] = frame.pixel_bytes;
    }

    memcpy(engine->render_frame_buffers[output_index], frame.pixels, frame.pixel_bytes);
    output->has_frame = 1;
    output->frame_width_px = (int)frame.width_px;
    output->frame_height_px = (int)frame.height_px;
    output->frame_channels = (int)frame.channels;
    output->frame_stride_bytes = (int)frame.stride_bytes;
    output->frame_byte_count = frame.pixel_bytes;
    output->frame_rgb = engine->render_frame_buffers[output_index];
    liftoff_render_release_frame(engine->render_engine, &frame);
}

void sim_engine_init(SimEngine* engine, PursuerParams params, State pursuer_initial) {
    pursuer_sim_init(&engine->pursuer, params, pursuer_initial);
    engine->num_targets = 0;
    engine->num_cameras = 0;
    engine->t = 0.0f;
    engine->intercept_radius_m = 0.0f;
    engine->metrics = empty_metrics();
    sim_engine_init_render_fields(engine);
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

LiftoffRenderStatus sim_engine_configure_rendering(
    SimEngine* engine,
    int enabled,
    int camera_id,
    int fail_on_error,
    const LiftoffRenderConfig* config
) {
    sim_engine_close_rendering(engine);
    engine->render_enabled = enabled ? 1 : 0;
    engine->render_camera_id = camera_id;
    engine->render_fail_on_error = fail_on_error ? 1 : 0;
    engine->render_sequence_id = 0;
    if (!engine->render_enabled) {
        return LIFTOFF_RENDER_DISABLED;
    }
    if (config == NULL) {
        engine->render_enabled = 0;
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }
    memcpy(&engine->render_config, config, sizeof(LiftoffRenderConfig));
    LiftoffRenderStatus status = liftoff_render_engine_create(
        &engine->render_config,
        &engine->render_engine
    );
    if (status != LIFTOFF_RENDER_OK) {
        engine->render_enabled = 0;
    }
    return status;
}

void sim_engine_close_rendering(SimEngine* engine) {
    if (engine == NULL) return;
    if (engine->render_engine != NULL) {
        liftoff_render_engine_destroy(engine->render_engine);
        engine->render_engine = NULL;
    }
    sim_engine_free_render_buffers(engine);
    engine->render_enabled = 0;
    engine->render_camera_id = -1;
    engine->render_fail_on_error = 0;
    engine->render_sequence_id = 0;
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
        output.frame_stride_bytes = camera->intrinsics.width_px * 3;
        output.frame_byte_count = 0;
        output.render_status = -1;
        output.frame_rgb = NULL;

        if (engine->num_targets <= 0) {
            output.observation.camera_id = camera->id;
            output.observation.target_index = -1;
            output.observation.captured = 1;
            output.observation.detected = 0;
            output.observation.t_capture = engine->t;
            sim_engine_render_camera_output(engine, camera, &output, count);
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
        sim_engine_render_camera_output(engine, camera, &output, count);
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

void sim_engine_get_snapshot(SimEngine* engine, SimSnapshot* snapshot) {
    if (engine == NULL || snapshot == NULL) return;
    memset(snapshot, 0, sizeof(SimSnapshot));

    snapshot->t = engine->t;
    snapshot->pursuer_state = engine->pursuer.state;
    snapshot->num_targets = engine->num_targets;
    if (snapshot->num_targets > SIM_MAX_TARGETS) {
        snapshot->num_targets = SIM_MAX_TARGETS;
    }
    for (int i = 0; i < snapshot->num_targets; i++) {
        snapshot->target_states[i] = target_sim_get_state(&engine->targets[i]);
        snapshot->target_ids[i] = engine->targets[i].id;
        snapshot->target_radii_m[i] = engine->targets[i].radius;
    }
    snapshot->intercept_radius_m = engine->intercept_radius_m;
    snapshot->metrics = engine->metrics;
    snapshot->num_camera_outputs = sim_engine_collect_camera_outputs(
        engine,
        snapshot->camera_outputs,
        SIM_MAX_CAMERA_OUTPUTS
    );
}

void sim_engine_batch_step_motor_speeds_dt(
    SimEngine* engines,
    const float* cmd_rpms,
    int num_engines,
    float dt,
    int substeps
) {
    if (engines == NULL || cmd_rpms == NULL || num_engines <= 0) return;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_engines; i++) {
        float speeds[4] = {
            cmd_rpms[i * 4 + 0],
            cmd_rpms[i * 4 + 1],
            cmd_rpms[i * 4 + 2],
            cmd_rpms[i * 4 + 3],
        };
        sim_engine_step_motor_speeds_dt(&engines[i], speeds, dt, substeps);
    }
}

void sim_engine_batch_get_snapshots(
    SimEngine* engines,
    int num_engines,
    SimSnapshots* snapshots
) {
    if (engines == NULL || snapshots == NULL || num_engines <= 0) return;
    snapshots->num_engines = num_engines;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_engines; i++) {
        SimEngine* engine = &engines[i];

        if (snapshots->pursuer_state != NULL) {
            State state = engine->pursuer.state;
            float* out = &snapshots->pursuer_state[i * SIM_SNAPSHOT_PURSUER_SIZE];
            out[0] = state.pos.x;
            out[1] = state.pos.y;
            out[2] = state.pos.z;
            out[3] = state.vel.x;
            out[4] = state.vel.y;
            out[5] = state.vel.z;
            out[6] = state.quat.x;
            out[7] = state.quat.y;
            out[8] = state.quat.z;
            out[9] = state.quat.w;
            out[10] = state.omega.x;
            out[11] = state.omega.y;
            out[12] = state.omega.z;
            out[13] = state.rpms[0];
            out[14] = state.rpms[1];
            out[15] = state.rpms[2];
            out[16] = state.rpms[3];
        }

        if (snapshots->first_target_state != NULL) {
            float* out = &snapshots->first_target_state[i * SIM_SNAPSHOT_TARGET_SIZE];
            if (engine->num_targets > 0) {
                TargetState target = target_sim_get_state(&engine->targets[0]);
                out[0] = target.pos.x;
                out[1] = target.pos.y;
                out[2] = target.pos.z;
                out[3] = target.vel.x;
                out[4] = target.vel.y;
                out[5] = target.vel.z;
            } else {
                for (int j = 0; j < 6; j++) out[j] = 0.0f;
            }
        }

        if (snapshots->metrics != NULL) {
            float* out = &snapshots->metrics[i * SIM_SNAPSHOT_METRICS_SIZE];
            out[0] = engine->metrics.distance_m;
            out[1] = engine->metrics.min_distance_m;
            out[2] = (float)engine->metrics.intercepted;
            out[3] = engine->metrics.intercept_time_s;
            out[4] = (float)engine->metrics.target_index;
        }

        if (snapshots->first_camera_observation != NULL) {
            float* out = &snapshots->first_camera_observation[i * SIM_SNAPSHOT_CAMERA_SIZE];
            out[0] = 0.0f;
            out[1] = 0.0f;
            out[2] = 0.0f;
            if (engine->num_cameras > 0 && engine->num_targets > 0) {
                CameraSim* camera = &engine->cameras[0];
                if (camera_sim_capture_due(camera, engine->t)) {
                    CameraObservation obs = camera_sim_observe_target(
                        camera,
                        &engine->pursuer.state,
                        &engine->targets[0],
                        0,
                        engine->t
                    );
                    out[0] = (float)obs.detected;
                    out[1] = obs.detected ? obs.uv_norm[0] : 0.0f;
                    out[2] = obs.detected ? obs.uv_norm[1] : 0.0f;
                }
            }
        }

        if (snapshots->max_rate_rps != NULL) {
            snapshots->max_rate_rps[i] = engine->pursuer.params.max_omega;
        }
        if (snapshots->max_rpm != NULL) {
            snapshots->max_rpm[i] = engine->pursuer.params.max_rpm;
        }
    }
}
