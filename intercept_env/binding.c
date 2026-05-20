#include "drone.h"
#include "render.h"

// Must match DRONE_OBS_SIZE in drone.h.
#define OBS_SIZE 26
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;
    env->task = (int)dict_get(kwargs, "task")->value;
    env->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    env->alpha_dist = dict_get(kwargs, "alpha_dist")->value;
    env->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    env->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    env->alpha_omega = dict_get(kwargs, "alpha_omega")->value;
    env->hover_target_dist = dict_get(kwargs, "hover_target_dist")->value;
    env->hover_dist = dict_get(kwargs, "hover_dist")->value;
    env->hover_omega = dict_get(kwargs, "hover_omega")->value;
    env->hover_vel = dict_get(kwargs, "hover_vel")->value;

    // Intercept task parameters (Thales/Gavin 2026).
    env->r_intercept = dict_get(kwargs, "r_intercept")->value;
    env->evader_dist_min = dict_get(kwargs, "evader_dist_min")->value;
    env->evader_dist_max = dict_get(kwargs, "evader_dist_max")->value;
    env->evader_speed_min = dict_get(kwargs, "evader_speed_min")->value;
    env->evader_speed_max = dict_get(kwargs, "evader_speed_max")->value;
    env->lambda_catch = dict_get(kwargs, "lambda_catch")->value;
    env->lambda_dist = dict_get(kwargs, "lambda_dist")->value;
    env->lambda_fail = dict_get(kwargs, "lambda_fail")->value;
    env->lambda_cmd = dict_get(kwargs, "lambda_cmd")->value;

    init(env);
}

void my_log(Log* log, Dict* out) {
    // score is incremented per-episode by hover_score (which for INTERCEPT is
    // the catch indicator). The dashboard reports score / n, i.e. catch_rate.
    dict_set(out, "catch_rate", log->score);
    // oob accumulates per failed episode; same denominator → fail_rate.
    dict_set(out, "fail_rate", log->oob);

    // legacy aliases kept for compatibility with hover/race tasks
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "rings_passed", log->rings_passed);
    dict_set(out, "ring_collisions", log->ring_collision);
    dict_set(out, "collisions", log->collisions);
    dict_set(out, "oob", log->oob);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "ema_dist", log->ema_dist);
    dict_set(out, "ema_vel", log->ema_vel);
    dict_set(out, "ema_omega", log->ema_omega);
}
