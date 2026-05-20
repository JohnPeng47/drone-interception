// Shared C simulation data types.

#pragma once

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos;      // global position (x, y, z)
    Vec3 vel;      // linear velocity (u, v, w)
    Quat quat;     // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega;    // angular velocity (p, q, r)
    float rpms[4]; // motor RPMs
} State;

typedef struct {
    Vec3 vel;         // Derivative of position
    Vec3 v_dot;       // Derivative of velocity
    Quat q_dot;       // Derivative of quaternion
    Vec3 w_dot;       // Derivative of angular velocity
    float rpm_dot[4]; // Derivative of motor RPMs
} StateDerivative;

typedef struct {
    float mass;       // kg
    float ixx;        // kgm^2
    float iyy;        // kgm^2
    float izz;        // kgm^2
    float arm_len;    // m
    float k_thrust;   // thrust coefficient (T = k * rpm^2)
    float k_ang_damp; // angular damping coefficient
    float k_drag;     // yaw moment constant (torque-to-thrust ratio style)
    float b_drag;     // linear drag coefficient
    float gravity;    // m/s^2 (positive, world gravity points -z)
    float max_rpm;    // RPM
    float max_vel;    // m/s (observation clamp)
    float max_omega;  // rad/s (observation clamp)
    float k_mot;      // s (motor RPM time constant)
    float rotor_pos_x[4]; // body-frame rotor x positions, optional
    float rotor_pos_y[4]; // body-frame rotor y positions, optional
    float rotor_dir[4];   // yaw moment signs, optional
} Params;
