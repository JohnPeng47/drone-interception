#include "sim_core.h"

#include "sim_math.h"

#define DT 0.002f
#define ACTION_SUBSTEPS 5
#define ACTION_DT (DT * (float)ACTION_SUBSTEPS)

static int has_rotor_geometry(const PursuerParams* params);

static inline float rpm_min_for_centered_hover(const PursuerParams* p) {
    // choose min_rpm so that action=0 -> (min+max)/2 == hover
    float hover = sqrtf((p->mass * p->gravity) / (4.0f * p->k_thrust));
    float min_rpm = 2.0f * hover - p->max_rpm;
    if (min_rpm < 0.0f) min_rpm = 0.0f;
    if (min_rpm > p->max_rpm) min_rpm = p->max_rpm;
    return min_rpm;
}

static void compute_derivatives(State* state, PursuerParams* params, float* actions,
                                StateDerivative* derivatives) {
    float min_rpm = rpm_min_for_centered_hover(params);

    float target_rpms[4];
    for (int i = 0; i < 4; i++) {
        float u = (actions[i] + 1.0f) * 0.5f; // [0,1]
        target_rpms[i] = min_rpm + u * (params->max_rpm - min_rpm);
    }

    float rpm_dot[4];
    for (int i = 0; i < 4; i++) {
        rpm_dot[i] = (1.0f / params->k_mot) * (target_rpms[i] - state->rpms[i]);
    }

    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = state->rpms[i];
        if (rpm < 0.0f) rpm = 0.0f;
        T[i] = params->k_thrust * rpm * rpm;
    }

    Vec3 F_prop_body = (Vec3){0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};
    Vec3 F_prop = quat_rotate(state->quat, F_prop_body);

    Vec3 F_aero;
    F_aero.x = -params->b_drag * state->vel.x;
    F_aero.y = -params->b_drag * state->vel.y;
    F_aero.z = -params->b_drag * state->vel.z;

    Vec3 v_dot;
    v_dot.x = (F_prop.x + F_aero.x) / params->mass;
    v_dot.y = (F_prop.y + F_aero.y) / params->mass;
    v_dot.z = ((F_prop.z + F_aero.z) / params->mass) - params->gravity;

    Quat omega_q = (Quat){0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    Vec3 Tau_prop;
    if (has_rotor_geometry(params)) {
        Tau_prop = (Vec3){0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 4; i++) {
            // Match rotorpy.compute_body_wrench for aero=False.
            Tau_prop.x += params->rotor_pos_y[i] * T[i];
            Tau_prop.y += -params->rotor_pos_x[i] * T[i];
            Tau_prop.z += params->k_drag * params->rotor_dir[i] * T[i];
        }
    } else {
        float arm_factor = params->arm_len / sqrtf(2.0f);
        Tau_prop.x = arm_factor * ((T[2] + T[3]) - (T[0] + T[1]));
        Tau_prop.y = arm_factor * ((T[1] + T[2]) - (T[0] + T[3]));
        Tau_prop.z = params->k_drag * (-T[0] + T[1] - T[2] + T[3]);
    }

    Vec3 Tau_aero;
    Tau_aero.x = -params->k_ang_damp * state->omega.x;
    Tau_aero.y = -params->k_ang_damp * state->omega.y;
    Tau_aero.z = -params->k_ang_damp * state->omega.z;

    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    Vec3 w_dot;
    w_dot.x = (Tau_prop.x + Tau_aero.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau_prop.y + Tau_aero.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau_prop.z + Tau_aero.z + Tau_iner.z) / params->izz;

    derivatives->vel = state->vel;
    derivatives->v_dot = v_dot;
    derivatives->q_dot = q_dot;
    derivatives->w_dot = w_dot;
    for (int i = 0; i < 4; i++) {
        derivatives->rpm_dot[i] = rpm_dot[i];
    }
}

static void rk4_step(State* state, PursuerParams* params, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    State temp_state;

    compute_derivatives(state, params, actions, &k1);

    step(state, &k1, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k2);

    step(state, &k2, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k3);

    step(state, &k3, dt, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    for (int i = 0; i < 4; i++) {
        state->rpms[i] +=
            (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt_6;
    }

    quat_normalize(&state->quat);
}

static int has_rotor_geometry(const PursuerParams* params) {
    for (int i = 0; i < 4; i++) {
        if (fabsf(params->rotor_pos_x[i]) > 1e-9f || fabsf(params->rotor_pos_y[i]) > 1e-9f) {
            return 1;
        }
    }
    return 0;
}

static void compute_derivatives_cmd_rpms(State* state, PursuerParams* params, float* cmd_rpms,
                                         StateDerivative* derivatives) {
    float rpm_dot[4];
    for (int i = 0; i < 4; i++) {
        float target = clampf(cmd_rpms[i], 0.0f, params->max_rpm);
        rpm_dot[i] = (1.0f / params->k_mot) * (target - state->rpms[i]);
    }

    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = state->rpms[i];
        if (rpm < 0.0f) rpm = 0.0f;
        T[i] = params->k_thrust * rpm * rpm;
    }

    Vec3 F_prop_body = (Vec3){0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};
    Vec3 F_prop = quat_rotate(state->quat, F_prop_body);

    Vec3 F_aero;
    F_aero.x = -params->b_drag * state->vel.x;
    F_aero.y = -params->b_drag * state->vel.y;
    F_aero.z = -params->b_drag * state->vel.z;

    Vec3 v_dot;
    v_dot.x = (F_prop.x + F_aero.x) / params->mass;
    v_dot.y = (F_prop.y + F_aero.y) / params->mass;
    v_dot.z = ((F_prop.z + F_aero.z) / params->mass) - params->gravity;

    Quat omega_q = (Quat){0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    Vec3 Tau_prop = (Vec3){0.0f, 0.0f, 0.0f};
    if (has_rotor_geometry(params)) {
        for (int i = 0; i < 4; i++) {
            // Match rotorpy.compute_body_wrench for aero=False.
            Tau_prop.x += params->rotor_pos_y[i] * T[i];
            Tau_prop.y += -params->rotor_pos_x[i] * T[i];
            Tau_prop.z += params->k_drag * params->rotor_dir[i] * T[i];
        }
    } else {
        float arm_factor = params->arm_len / sqrtf(2.0f);
        Tau_prop.x = arm_factor * ((T[2] + T[3]) - (T[0] + T[1]));
        Tau_prop.y = arm_factor * ((T[1] + T[2]) - (T[0] + T[3]));
        Tau_prop.z = params->k_drag * (-T[0] + T[1] - T[2] + T[3]);
    }

    Vec3 Tau_aero;
    Tau_aero.x = -params->k_ang_damp * state->omega.x;
    Tau_aero.y = -params->k_ang_damp * state->omega.y;
    Tau_aero.z = -params->k_ang_damp * state->omega.z;

    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    Vec3 w_dot;
    w_dot.x = (Tau_prop.x + Tau_aero.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau_prop.y + Tau_aero.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau_prop.z + Tau_aero.z + Tau_iner.z) / params->izz;

    derivatives->vel = state->vel;
    derivatives->v_dot = v_dot;
    derivatives->q_dot = q_dot;
    derivatives->w_dot = w_dot;
    for (int i = 0; i < 4; i++) {
        derivatives->rpm_dot[i] = rpm_dot[i];
    }
}

static void rk4_step_cmd_rpms(State* state, PursuerParams* params, float* cmd_rpms, float dt) {
    StateDerivative k1, k2, k3, k4;
    State temp_state;

    compute_derivatives_cmd_rpms(state, params, cmd_rpms, &k1);

    step(state, &k1, dt * 0.5f, &temp_state);
    compute_derivatives_cmd_rpms(&temp_state, params, cmd_rpms, &k2);

    step(state, &k2, dt * 0.5f, &temp_state);
    compute_derivatives_cmd_rpms(&temp_state, params, cmd_rpms, &k3);

    step(state, &k3, dt, &temp_state);
    compute_derivatives_cmd_rpms(&temp_state, params, cmd_rpms, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    for (int i = 0; i < 4; i++) {
        state->rpms[i] +=
            (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt_6;
    }

    quat_normalize(&state->quat);
}

void pursuer_sim_init(PursuerSim* sim, PursuerParams params, State initial) {
    sim->params = params;
    sim->state = initial;
    quat_normalize(&sim->state.quat);
}

void pursuer_sim_reset(PursuerSim* sim, State initial) {
    sim->state = initial;
    quat_normalize(&sim->state.quat);
}

void pursuer_sim_step_motor(PursuerSim* sim, float actions[4]) {
    pursuer_sim_step_motor_dt(sim, actions, ACTION_DT, ACTION_SUBSTEPS);
}

void pursuer_sim_step_motor_dt(PursuerSim* sim, float actions[4], float dt, int substeps) {
    clamp4(actions, -1.0f, 1.0f);
    if (substeps < 1) substeps = 1;
    float sub_dt = dt / (float)substeps;
    for (int s = 0; s < substeps; s++) {
        rk4_step(&sim->state, &sim->params, actions, sub_dt);
        clamp3(&sim->state.vel, -sim->params.max_vel, sim->params.max_vel);
        clamp3(&sim->state.omega, -sim->params.max_omega, sim->params.max_omega);
        for (int i = 0; i < 4; i++) {
            sim->state.rpms[i] = clampf(sim->state.rpms[i], 0.0f, sim->params.max_rpm);
        }
    }
}

void pursuer_sim_step_motor_speeds_dt(PursuerSim* sim, float cmd_rpms[4], float dt, int substeps) {
    if (substeps < 1) substeps = 1;
    float sub_dt = dt / (float)substeps;
    for (int s = 0; s < substeps; s++) {
        rk4_step_cmd_rpms(&sim->state, &sim->params, cmd_rpms, sub_dt);
        clamp3(&sim->state.vel, -sim->params.max_vel, sim->params.max_vel);
        clamp3(&sim->state.omega, -sim->params.max_omega, sim->params.max_omega);
        for (int i = 0; i < 4; i++) {
            sim->state.rpms[i] = clampf(sim->state.rpms[i], 0.0f, sim->params.max_rpm);
        }
    }
}

State pursuer_sim_get_state(PursuerSim* sim) {
    return sim->state;
}
