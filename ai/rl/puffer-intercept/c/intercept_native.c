#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sim_engine.h"
#include "sim_math.h"

#define CSIM_MAGIC "CSIMINST"
#define CSIM_VERSION 8
#define OBS_SIZE 26

typedef struct {
    float catch_reward;
    float distance_weight;
    float progress_weight;
    float fail_penalty;
    float rate_weight;
} NativeRewardConfig;

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

typedef struct {
    const unsigned char* data;
    size_t size;
    size_t offset;
} Cursor;

static int cursor_read(Cursor* cursor, void* out, size_t size) {
    if (cursor->offset + size > cursor->size) return 0;
    memcpy(out, cursor->data + cursor->offset, size);
    cursor->offset += size;
    return 1;
}

static int read_u8(Cursor* cursor, uint8_t* out) {
    return cursor_read(cursor, out, sizeof(*out));
}

static int read_u16(Cursor* cursor, uint16_t* out) {
    return cursor_read(cursor, out, sizeof(*out));
}

static int read_u32(Cursor* cursor, uint32_t* out) {
    return cursor_read(cursor, out, sizeof(*out));
}

static int read_i64(Cursor* cursor, int64_t* out) {
    return cursor_read(cursor, out, sizeof(*out));
}

static int read_f32(Cursor* cursor, float* out) {
    return cursor_read(cursor, out, sizeof(*out));
}

static int skip_bytes(Cursor* cursor, size_t size) {
    if (cursor->offset + size > cursor->size) return 0;
    cursor->offset += size;
    return 1;
}

static int skip_string(Cursor* cursor) {
    uint16_t len = 0;
    if (!read_u16(cursor, &len)) return 0;
    return skip_bytes(cursor, len);
}

static int skip_optional_string(Cursor* cursor) {
    uint8_t present = 0;
    if (!read_u8(cursor, &present)) return 0;
    return present ? skip_string(cursor) : 1;
}

static int read_vec3(Cursor* cursor, Vec3* out) {
    return read_f32(cursor, &out->x) && read_f32(cursor, &out->y) && read_f32(cursor, &out->z);
}

static int skip_f32_array(Cursor* cursor, int count) {
    return skip_bytes(cursor, (size_t)count * sizeof(float));
}

static int read_optional_f32_array(Cursor* cursor, float* out, int count) {
    uint8_t present = 0;
    if (!read_u8(cursor, &present)) return 0;
    if (!present) return 1;
    if (out == NULL) return skip_f32_array(cursor, count);
    return cursor_read(cursor, out, (size_t)count * sizeof(float));
}

static float hover_rpm(const PursuerParams* params) {
    float denom = fmaxf(4.0f * params->k_thrust, 1e-12f);
    return sqrtf((params->mass * params->gravity) / denom);
}

static float default_min_rpm(const PursuerParams* params) {
    float rpm = 2.0f * hover_rpm(params) - params->max_rpm;
    return clampf(rpm, 0.0f, params->max_rpm);
}

static int read_pursuer_params(Cursor* cursor, PursuerParams* params, float* rpm_min, float* k_w) {
    memset(params, 0, sizeof(*params));
    if (!read_f32(cursor, &params->mass)) return 0;
    if (!read_f32(cursor, &params->ixx)) return 0;
    if (!read_f32(cursor, &params->iyy)) return 0;
    if (!read_f32(cursor, &params->izz)) return 0;
    if (!read_f32(cursor, &params->arm_len)) return 0;
    if (!read_f32(cursor, &params->k_thrust)) return 0;
    if (!read_f32(cursor, &params->k_drag)) return 0;
    if (!read_f32(cursor, &params->k_ang_damp)) return 0;
    if (!read_f32(cursor, &params->b_drag)) return 0;
    if (!read_f32(cursor, &params->gravity)) return 0;
    if (!read_f32(cursor, &params->max_rpm)) return 0;
    if (!read_f32(cursor, &params->max_vel)) return 0;
    if (!read_f32(cursor, &params->max_omega)) return 0;
    if (!read_f32(cursor, &params->k_mot)) return 0;
    if (!read_f32(cursor, k_w)) return 0;

    uint8_t has_rpm_min = 0;
    if (!read_u8(cursor, &has_rpm_min)) return 0;
    if (has_rpm_min) {
        if (!read_f32(cursor, rpm_min)) return 0;
    } else {
        *rpm_min = default_min_rpm(params);
    }

    float positions[12] = {0};
    if (!read_optional_f32_array(cursor, positions, 12)) return 0;
    for (int i = 0; i < 4; i++) {
        params->rotor_pos_x[i] = positions[i * 3 + 0];
        params->rotor_pos_y[i] = positions[i * 3 + 1];
    }
    if (!read_optional_f32_array(cursor, params->rotor_dir, 4)) return 0;
    return 1;
}

static int read_pursuer_initial(Cursor* cursor, State* state, const PursuerParams* params) {
    float q_xyzw[4] = {0};
    memset(state, 0, sizeof(*state));
    if (!read_vec3(cursor, &state->pos)) return 0;
    if (!read_vec3(cursor, &state->vel)) return 0;
    if (!cursor_read(cursor, q_xyzw, sizeof(q_xyzw))) return 0;
    state->quat = (Quat){q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]};
    quat_normalize(&state->quat);
    if (!read_vec3(cursor, &state->omega)) return 0;

    uint8_t has_rpms = 0;
    if (!read_u8(cursor, &has_rpms)) return 0;
    if (has_rpms) {
        if (!cursor_read(cursor, state->rpms, 4 * sizeof(float))) return 0;
    } else {
        float rpm = hover_rpm(params);
        for (int i = 0; i < 4; i++) state->rpms[i] = rpm;
    }
    return read_optional_f32_array(cursor, NULL, 3);
}

static int read_target_initial(Cursor* cursor, TargetState* state) {
    return read_vec3(cursor, &state->pos) && read_vec3(cursor, &state->vel);
}

static int read_target_config(Cursor* cursor, int id, const TargetState* initial, TargetSim* target) {
    if (!skip_string(cursor)) return 0;
    if (!skip_string(cursor)) return 0;
    float radius = 0.0f;
    if (!read_f32(cursor, &radius)) return 0;

    if (!skip_string(cursor)) return 0;
    uint16_t waypoint_count = 0;
    if (!read_u16(cursor, &waypoint_count)) return 0;
    TargetBehaviorConfig behavior = {0};
    behavior.kind = TARGET_BEHAVIOR_WAYPOINTS;
    behavior.num_waypoints = waypoint_count > SIM_MAX_WAYPOINTS ? SIM_MAX_WAYPOINTS : waypoint_count;
    for (int i = 0; i < waypoint_count; i++) {
        Vec3 waypoint = {0};
        if (!read_vec3(cursor, &waypoint)) return 0;
        if (i < SIM_MAX_WAYPOINTS) behavior.waypoints[i] = waypoint;
    }
    if (behavior.num_waypoints == 0) {
        behavior.num_waypoints = 1;
        behavior.waypoints[0] = initial->pos;
    }
    if (!read_f32(cursor, &behavior.duration)) return 0;
    uint8_t loop = 0;
    if (!read_u8(cursor, &loop)) return 0;
    behavior.loop = loop ? 1 : 0;

    if (!skip_string(cursor)) return 0;
    TargetControllerConfig controller = {0};
    controller.kind = TARGET_CONTROLLER_LINEAR;
    if (!read_f32(cursor, &controller.kp)) return 0;
    if (!read_f32(cursor, &controller.kv)) return 0;
    if (!read_f32(cursor, &controller.max_accel)) return 0;

    target_sim_init(target, id, radius, *initial, behavior, controller);
    return 1;
}

static int read_camera_config(Cursor* cursor, int id, CameraSim* camera) {
    memset(camera, 0, sizeof(*camera));
    camera->id = id;
    if (!skip_string(cursor)) return 0;
    if (!skip_string(cursor)) return 0;
    if (!read_vec3(cursor, &camera->position_b)) return 0;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            if (!read_f32(cursor, &camera->body_to_camera.m[r][c])) return 0;
        }
    }
    uint32_t width = 0, height = 0;
    if (!read_u32(cursor, &width)) return 0;
    if (!read_u32(cursor, &height)) return 0;
    camera->intrinsics.width_px = (int)width;
    camera->intrinsics.height_px = (int)height;
    if (!read_f32(cursor, &camera->intrinsics.fx_px)) return 0;
    if (!read_f32(cursor, &camera->intrinsics.fy_px)) return 0;
    if (!read_f32(cursor, &camera->intrinsics.cx_px)) return 0;
    if (!read_f32(cursor, &camera->intrinsics.cy_px)) return 0;
    if (!read_f32(cursor, &camera->intrinsics.hfov_rad)) return 0;
    if (!read_f32(cursor, &camera->intrinsics.vfov_rad)) return 0;
    if (!read_f32(cursor, &camera->capture_rate_hz)) return 0;
    camera_sim_reset(camera);
    return 1;
}

static int skip_noise_config(Cursor* cursor) {
    return skip_f32_array(cursor, 10) && skip_bytes(cursor, sizeof(int64_t));
}

static int skip_render_config(Cursor* cursor) {
    uint32_t timeout = 0;
    uint8_t fail = 0;
    return skip_optional_string(cursor) &&
        skip_string(cursor) &&
        skip_string(cursor) &&
        skip_string(cursor) &&
        read_u32(cursor, &timeout) &&
        read_u8(cursor, &fail);
}

static int read_sim_config(Cursor* cursor, NativeScenario* scenario, const TargetState* target_initials, int target_initial_count) {
    uint8_t present = 0;
    if (!read_u8(cursor, &present)) return 0;
    if (!present) return 0;

    if (!read_pursuer_params(cursor, &scenario->pursuer_params, &scenario->rpm_min, &scenario->k_w)) return 0;
    if (!read_f32(cursor, &scenario->dt)) return 0;
    uint32_t substeps = 0;
    if (!read_u32(cursor, &substeps)) return 0;
    scenario->substeps = substeps > 0 ? (int)substeps : 1;
    scenario->dt *= (float)scenario->substeps;
    if (!read_f32(cursor, &scenario->duration_s)) return 0;
    uint8_t has_validation_dt = 0;
    if (!read_u8(cursor, &has_validation_dt)) return 0;
    if (has_validation_dt && !skip_f32_array(cursor, 1)) return 0;
    if (!skip_string(cursor)) return 0;
    if (!skip_f32_array(cursor, 1)) return 0;
    uint8_t randomize = 0;
    if (!read_u8(cursor, &randomize)) return 0;

    uint16_t target_count = 0;
    if (!read_u16(cursor, &target_count)) return 0;
    scenario->num_targets = target_count > SIM_MAX_TARGETS ? SIM_MAX_TARGETS : target_count;
    for (int i = 0; i < target_count; i++) {
        TargetState initial = i < target_initial_count ? target_initials[i] : (TargetState){0};
        TargetSim target = {0};
        if (!read_target_config(cursor, i, &initial, &target)) return 0;
        if (i < SIM_MAX_TARGETS) scenario->targets[i] = target;
    }

    uint16_t camera_count = 0;
    if (!read_u16(cursor, &camera_count)) return 0;
    scenario->num_cameras = camera_count > SIM_MAX_CAMERAS ? SIM_MAX_CAMERAS : camera_count;
    for (int i = 0; i < camera_count; i++) {
        CameraSim camera = {0};
        if (!read_camera_config(cursor, i, &camera)) return 0;
        if (i < SIM_MAX_CAMERAS) scenario->cameras[i] = camera;
    }

    if (!read_f32(cursor, &scenario->intercept_radius_m)) return 0;
    if (!read_f32(cursor, &scenario->max_thrust_n)) return 0;
    if (!read_f32(cursor, &scenario->max_rate_rps)) return 0;
    uint8_t has_bounds = 0;
    if (!read_u8(cursor, &has_bounds)) return 0;
    scenario->has_bounds = has_bounds ? 1 : 0;
    if (has_bounds && !read_vec3(cursor, &scenario->bounds_w)) return 0;
    if (!skip_noise_config(cursor)) return 0;
    uint8_t rendering = 0;
    if (!read_u8(cursor, &rendering)) return 0;
    return skip_render_config(cursor);
}

static int read_scenario(Cursor* cursor, NativeScenario* scenario) {
    memset(scenario, 0, sizeof(*scenario));
    if (!read_i64(cursor, &scenario->seed)) return 0;

    size_t pursuer_offset = cursor->offset;
    State temp_state = {0};
    PursuerParams default_params = {
        .mass = 1.0f,
        .gravity = 9.81f,
        .k_thrust = 1.0f,
        .max_rpm = 1.0f,
    };
    if (!read_pursuer_initial(cursor, &temp_state, &default_params)) return 0;

    uint16_t target_initial_count = 0;
    if (!read_u16(cursor, &target_initial_count)) return 0;
    TargetState target_initials[SIM_MAX_TARGETS];
    for (int i = 0; i < target_initial_count; i++) {
        TargetState target_initial = {0};
        if (!read_target_initial(cursor, &target_initial)) return 0;
        if (i < SIM_MAX_TARGETS) target_initials[i] = target_initial;
    }
    if (!read_sim_config(cursor, scenario, target_initials, target_initial_count)) return 0;

    Cursor pursuer_cursor = *cursor;
    pursuer_cursor.offset = pursuer_offset;
    return read_pursuer_initial(&pursuer_cursor, &scenario->pursuer_initial, &scenario->pursuer_params);
}

static int load_scenarios(const char* path, NativeScenario** scenarios_out, int* count_out) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) return 0;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }
    long file_size_long = ftell(file);
    if (file_size_long <= 0) {
        fclose(file);
        return 0;
    }
    rewind(file);
    size_t file_size = (size_t)file_size_long;
    unsigned char* data = (unsigned char*)malloc(file_size);
    if (data == NULL) {
        fclose(file);
        return 0;
    }
    size_t read_count = fread(data, 1, file_size, file);
    fclose(file);
    if (read_count != file_size) {
        free(data);
        return 0;
    }

    Cursor cursor = {.data = data, .size = file_size, .offset = 0};
    char magic[8];
    uint32_t version = 0;
    uint32_t count = 0;
    uint64_t payload_len = 0;
    int ok = cursor_read(&cursor, magic, sizeof(magic)) &&
        read_u32(&cursor, &version) &&
        read_u32(&cursor, &count) &&
        cursor_read(&cursor, &payload_len, sizeof(payload_len));
    if (!ok || memcmp(magic, CSIM_MAGIC, 8) != 0 || version != CSIM_VERSION) {
        free(data);
        return 0;
    }
    if (cursor.offset + (size_t)payload_len != file_size || count == 0) {
        free(data);
        return 0;
    }

    NativeScenario* scenarios = (NativeScenario*)calloc(count, sizeof(NativeScenario));
    if (scenarios == NULL) {
        free(data);
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!read_scenario(&cursor, &scenarios[i])) {
            free(scenarios);
            free(data);
            return 0;
        }
    }
    free(data);
    *scenarios_out = scenarios;
    *count_out = (int)count;
    return 1;
}

static void write_observation(SimEngine* engine, float max_rate, float max_rpm, float* out) {
    const State* pursuer = &engine->pursuer.state;
    TargetState target = engine->num_targets > 0 ? target_sim_get_state(&engine->targets[0]) : (TargetState){0};
    Vec3 rel_pos_w = sub3(target.pos, pursuer->pos);
    Vec3 rel_vel_w = sub3(target.vel, pursuer->vel);
    Vec3 vel_b = quat_rotate(quat_inverse(pursuer->quat), pursuer->vel);
    Vec3 rel_pos_b = quat_rotate(quat_inverse(pursuer->quat), rel_pos_w);
    Vec3 rel_vel_b = quat_rotate(quat_inverse(pursuer->quat), rel_vel_w);
    Vec3 gravity_b = quat_rotate(quat_inverse(pursuer->quat), (Vec3){0.0f, 0.0f, -1.0f});
    float uv0 = 0.0f;
    float uv1 = 0.0f;
    if (engine->num_cameras > 0 && engine->num_targets > 0 && camera_sim_capture_due(&engine->cameras[0], engine->t)) {
        CameraObservation obs = camera_sim_observe_target(&engine->cameras[0], pursuer, &engine->targets[0], 0, engine->t);
        if (obs.detected) {
            uv0 = obs.uv_norm[0];
            uv1 = obs.uv_norm[1];
        }
    }
    float range_norm = fmaxf(norm3(rel_pos_w), 1e-9f);
    float closing_speed = dot3(rel_vel_w, rel_pos_w) / range_norm;
    float vel_denom = 20.0f * sqrtf(3.0f);
    float safe_max_rate = fmaxf(max_rate, 1e-6f);
    float safe_max_rpm = fmaxf(max_rpm, 1e-6f);

    out[0] = vel_b.x / vel_denom;
    out[1] = vel_b.y / vel_denom;
    out[2] = vel_b.z / vel_denom;
    out[3] = pursuer->omega.x / safe_max_rate;
    out[4] = pursuer->omega.y / safe_max_rate;
    out[5] = pursuer->omega.z / safe_max_rate;
    out[6] = gravity_b.x;
    out[7] = gravity_b.y;
    out[8] = gravity_b.z;
    out[9] = tanhf(rel_pos_b.x * 0.1f);
    out[10] = tanhf(rel_pos_b.y * 0.1f);
    out[11] = tanhf(rel_pos_b.z * 0.1f);
    out[12] = tanhf(rel_pos_b.x * 10.0f);
    out[13] = tanhf(rel_pos_b.y * 10.0f);
    out[14] = tanhf(rel_pos_b.z * 10.0f);
    out[15] = rel_vel_b.x / vel_denom;
    out[16] = rel_vel_b.y / vel_denom;
    out[17] = rel_vel_b.z / vel_denom;
    out[18] = uv0;
    out[19] = uv1;
    out[20] = engine->metrics.distance_m / 20.0f;
    out[21] = closing_speed / 8.0f;
    out[22] = pursuer->rpms[0] / safe_max_rpm;
    out[23] = pursuer->rpms[1] / safe_max_rpm;
    out[24] = pursuer->rpms[2] / safe_max_rpm;
    out[25] = pursuer->rpms[3] / safe_max_rpm;
}
