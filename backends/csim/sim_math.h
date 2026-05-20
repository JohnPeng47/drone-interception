// Shared C simulation math helpers.

#pragma once

#include <math.h>
#include <stdlib.h>

#include "sim_types.h"

static inline float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static inline float rndf(float a, float b, unsigned int* rng) {
    return a + ((float)rand_r(rng) / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline Quat add_quat(Quat a, Quat b) {
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Quat scalmul_quat(Quat a, float b) {
    return (Quat){a.w * b, a.x * b, a.y * b, a.z * b};
}

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline void clamp3(Vec3* vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

static inline void clamp4(float a[4], float min, float max) {
    a[0] = clampf(a[0], min, max);
    a[1] = clampf(a[1], min, max);
    a[2] = clampf(a[2], min, max);
    a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline void quat_normalize(Quat* q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = (Quat){0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = (Quat){q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) { return (Quat){q.w, -q.x, -q.y, -q.z}; }

static inline Quat rndquat(unsigned int* rng) {
    float u1 = rndf(0.0f, 1.0f, rng);
    float u2 = rndf(0.0f, 1.0f, rng);
    float u3 = rndf(0.0f, 1.0f, rng);

    float sqrt_1_minus_u1 = sqrtf(1.0f - u1);
    float sqrt_u1 = sqrtf(u1);

    float pi_2_u2 = 2.0f * (float)M_PI * u2;
    float pi_2_u3 = 2.0f * (float)M_PI * u3;

    Quat q;
    q.w = sqrt_1_minus_u1 * sinf(pi_2_u2);
    q.x = sqrt_1_minus_u1 * cosf(pi_2_u2);
    q.y = sqrt_u1 * sinf(pi_2_u3);
    q.z = sqrt_u1 * cosf(pi_2_u3);
    return q;
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float half = angle * 0.5f;
    float s = sinf(half);
    return (Quat){cosf(half), axis.x * s, axis.y * s, axis.z * s};
}

static inline void step(State* initial, StateDerivative* deriv, float dt, State* output) {
    output->pos = add3(initial->pos, scalmul3(deriv->vel, dt));
    output->vel = add3(initial->vel, scalmul3(deriv->v_dot, dt));
    output->quat = add_quat(initial->quat, scalmul_quat(deriv->q_dot, dt));
    output->omega = add3(initial->omega, scalmul3(deriv->w_dot, dt));
    for (int i = 0; i < 4; i++) {
        output->rpms[i] = initial->rpms[i] + deriv->rpm_dot[i] * dt;
    }
    quat_normalize(&output->quat);
}
