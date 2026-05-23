#include "camera_sim.h"

#include "sim_math.h"

void camera_sim_reset(CameraSim* camera) {
    camera->next_capture_t = 0.0f;
}

int camera_sim_capture_due(CameraSim* camera, float t) {
    if (camera->capture_rate_hz <= 0.0f) return 0;
    if (t + 1e-12f < camera->next_capture_t) return 0;
    camera->next_capture_t += 1.0f / camera->capture_rate_hz;
    return 1;
}

CameraObservation camera_sim_observe_target(
    const CameraSim* camera,
    const State* pursuer,
    const TargetSim* target,
    int target_index,
    float t
) {
    Vec3 camera_pos_w = add3(
        pursuer->pos,
        quat_rotate(pursuer->quat, camera->position_b)
    );
    Vec3 target_delta_w = sub3(target->state.pos, camera_pos_w);
    Vec3 target_delta_b = quat_rotate(quat_inverse(pursuer->quat), target_delta_w);
    Vec3 target_pos_c = mat3_mul_vec3(camera->body_to_camera, target_delta_b);
    float range_m = norm3(sub3(target->state.pos, pursuer->pos));

    CameraObservation obs = {0};
    obs.camera_id = camera->id;
    obs.target_index = target_index;
    obs.captured = 1;
    obs.detected = 0;
    obs.t_capture = t;
    obs.target_pos_c = target_pos_c;
    obs.range_m = range_m;

    float forward = target_pos_c.x;
    if (forward <= 1e-9f) return obs;

    float u_norm = target_pos_c.y / forward;
    float v_norm = target_pos_c.z / forward;
    float tan_h = tanf(camera->intrinsics.hfov_rad * 0.5f);
    float tan_v = tanf(camera->intrinsics.vfov_rad * 0.5f);
    if (fabsf(u_norm) > tan_h || fabsf(v_norm) > tan_v) return obs;

    obs.detected = 1;
    obs.uv_norm[0] = u_norm;
    obs.uv_norm[1] = v_norm;
    obs.uv_px[0] = camera->intrinsics.fx_px * u_norm + camera->intrinsics.cx_px;
    obs.uv_px[1] = camera->intrinsics.fy_px * v_norm + camera->intrinsics.cy_px;
    obs.apparent_radius_px = camera->intrinsics.fx_px * target->radius / fmaxf(range_m, 1e-9f);
    return obs;
}
