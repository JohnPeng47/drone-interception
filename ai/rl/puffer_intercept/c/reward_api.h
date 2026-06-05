#pragma once

#include "native_types.h"
#include "sim_engine.h"

typedef struct {
    float values[16];
    int ints[16];
} NativeRewardState;

typedef struct {
    const SimEngine* engine;
    const NativeScenario* scenario;
    const float* action;
    const float* previous_action;
    const float* body_rates_b;
    float thrust_n;
    float max_thrust_n;
    float max_rate_rps;
    float previous_distance_m;
    float distance_m;
    float rate_norm;
    float elapsed_s;
    int episode_length;
    int intercepted;
    int failed;
    int timeout;
} NativeRewardStep;

void native_reward_reset(
    NativeRewardState* state,
    const SimEngine* engine,
    const NativeScenario* scenario
);

float native_reward_step(
    NativeRewardState* state,
    const NativeRewardStep* step
);
