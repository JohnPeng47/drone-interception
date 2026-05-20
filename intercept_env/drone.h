// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "dronelib.h"
#include "tasks.h"

#define HORIZON 1024

typedef struct Client Client;
typedef struct DroneEnv DroneEnv;

struct DroneEnv {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int tick;
    DroneTask task;
    Drone* agents;

    int max_rings;
    Target* ring_buffer;

    Client* client;

    // reward scaling
    float alpha_dist;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;

    // hover task parameters
    float hover_target_dist;
    float hover_dist;
    float hover_omega;
    float hover_vel;

    // intercept task parameters (Thales/Gavin 2026 paper Table I)
    float r_intercept;        // capture distance threshold (m)
    float evader_dist_min;    // initial pursuer-evader range (m)
    float evader_dist_max;
    float evader_speed_min;   // evader scalar speed (m/s)
    float evader_speed_max;
    float lambda_catch;       // paper: 10.0
    float lambda_dist;        // paper: 0.001
    float lambda_fail;        // paper: 30.0
    float lambda_cmd;         // paper: 2e-4
};

void init(DroneEnv* env) {
    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = (Target*)calloc(env->max_rings, sizeof(Target));

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));
        env->agents[i].buffer_idx = 0;
    }

    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneEnv* env, int idx, bool oob, bool timeout) {
    Drone* agent = &env->agents[idx];

    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collisions += agent->collisions;

    if (oob) env->log.oob += 1.0f;
    if (timeout) env->log.timeout += 1.0f;

    env->log.score += agent->hover_score;
    env->log.perf += agent->hover_ema;
    env->log.rings_passed += agent->rings_passed;
    env->log.ema_dist += agent->ema_dist;
    env->log.ema_vel += agent->ema_vel;
    env->log.ema_omega += agent->ema_omega;

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->rings_passed = 0.0f;
}

// 23 base + 3 for (v_target - v_self) in body frame = 26.
// Must match OBS_SIZE in binding.c.
#define DRONE_OBS_SIZE 26

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        compute_drone_observations(&env->agents[i], env->observations + i*DRONE_OBS_SIZE);
    }
}

void reset_agent(DroneEnv* env, Drone* agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->rings_passed = 0;
    agent->score = 0.0f;
    agent->hover_score = 0.0f;
    agent->hover_ema = 0.0f;
    agent->ema_dist = 0.0f;
    agent->ema_vel = 0.0f;
    agent->ema_omega = 0.0f;

    agent->buffer = env->ring_buffer;
    agent->buffer_size = env->max_rings;

    init_drone(agent, &env->rng, 0.05f);

    agent->state.pos =
        (Vec3){rndf(-MARGIN_X, MARGIN_X, &env->rng), rndf(-MARGIN_Y, MARGIN_Y, &env->rng), rndf(-MARGIN_Z, MARGIN_Z, &env->rng)};

    if (env->task == RACE) {
        while (norm3(sub3(agent->state.pos, env->ring_buffer[0].pos)) < 2.0f * RING_RADIUS) {
            agent->state.pos = (Vec3){rndf(-MARGIN_X, MARGIN_X, &env->rng), rndf(-MARGIN_Y, MARGIN_Y, &env->rng),
                                      rndf(-MARGIN_Z, MARGIN_Z, &env->rng)};
        }
    }

    agent->prev_pos = agent->state.pos;
    agent->prev_potential = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
}

static inline void place_target(DroneEnv* env, int i) {
    Drone* agent = &env->agents[i];
    if (env->task == INTERCEPT) {
        set_target_intercept(&env->rng, agent,
                             env->evader_dist_min, env->evader_dist_max,
                             env->evader_speed_min, env->evader_speed_max);
    } else {
        set_target(&env->rng, env->task, env->agents, i, env->num_agents,
                   env->hover_target_dist);
    }
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_rings(&env->rng, env->ring_buffer, env->max_rings);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent(env, agent, i);
        place_target(env, i);
    }

    compute_observations(env);
}

void c_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];

        agent->prev_pos = agent->state.pos;
        move_drone(agent, &env->actions[4 * i]);
        agent->episode_length++;

        if (env->task == INTERCEPT) {
            // Propagate kinematic evader one tick (vel is per-step displacement).
            move_target(agent);

            float dist = norm3(sub3(agent->target->pos, agent->state.pos));
            float omega = norm3(agent->state.omega);

            bool caught = (dist < env->r_intercept);
            bool oob = (fabsf(agent->state.pos.x) > MARGIN_X)
                     || (fabsf(agent->state.pos.y) > MARGIN_Y)
                     || (fabsf(agent->state.pos.z) > MARGIN_Z);
            bool timeout = (agent->episode_length >= HORIZON);
            bool fail = oob;  // ground crash is just z<-MARGIN_Z, covered by oob

            float reward = (caught ? env->lambda_catch : 0.0f)
                         - env->lambda_dist * dist
                         - (fail ? env->lambda_fail : 0.0f)
                         - env->lambda_cmd * omega;

            agent->ema_dist = 0.99f * agent->ema_dist + 0.01f * dist;
            agent->ema_vel = 0.99f * agent->ema_vel + 0.01f * norm3(agent->state.vel);
            agent->ema_omega = 0.99f * agent->ema_omega + 0.01f * omega;
            agent->episode_return += reward;
            env->rewards[i] = reward;

            bool reset = caught || fail || timeout;
            env->terminals[i] = reset ? 1.0f : 0.0f;

            if (caught) agent->hover_score += 1.0f;  // reuse field as catch counter

            if (reset) {
                add_log(env, i, oob, timeout);
                reset_agent(env, agent, i);
                place_target(env, i);
            }
        } else {
            bool oob = norm3(sub3(agent->target->pos, agent->state.pos)) > (env->hover_target_dist + 1.0f);
            bool timeout = (agent->episode_length >= HORIZON);

            float curr = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
            float prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
            float curr_dist = norm3(sub3(agent->target->pos, agent->state.pos));
            float omega = norm3(agent->state.omega);

            float reward = env->alpha_dist * (prev_dist - curr_dist)
                         + env->alpha_hover * curr
                         + env->alpha_shaping * (curr - agent->prev_potential)
                         - env->alpha_omega * omega;

            agent->prev_potential = curr;

            float h = check_hover(agent, env->hover_dist, env->hover_omega, env->hover_vel);
            agent->hover_score += h;
            agent->hover_ema = (1.0f - 0.02f) * agent->hover_ema + 0.02f * h;
            agent->ema_dist = 0.99f * agent->ema_dist + 0.01f * curr_dist;
            agent->ema_vel = 0.99f * agent->ema_vel + 0.01f * norm3(agent->state.vel);
            agent->ema_omega = 0.99f * agent->ema_omega + 0.01f * omega;
            agent->episode_return += reward;
            env->rewards[i] = reward;

            bool reset = oob || timeout;
            env->terminals[i] = reset ? 1.0f : 0.0f;

            if (reset) {
                add_log(env, i, oob, timeout);
                reset_agent(env, agent, i);
                place_target(env, i);
            }
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        free(env->agents[i].target);
    }

    free(env->agents);
    free(env->ring_buffer);

    if (env->client != NULL) {
        c_close_client(env->client);
    }
}
