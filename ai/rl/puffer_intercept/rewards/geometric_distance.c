#include <math.h>
#include <string.h>

#include "reward_api.h"

enum {
    PREVIOUS_DISTANCE_SLOT = 0,
};

void native_reward_reset(
    NativeRewardState* state,
    const SimEngine* engine,
    const NativeScenario* scenario
) {
    (void)scenario;
    memset(state, 0, sizeof(*state));
    state->values[PREVIOUS_DISTANCE_SLOT] = engine->metrics.distance_m;
}

float native_reward_step(
    NativeRewardState* state,
    const NativeRewardStep* step
) {
    const float progress_weight = 28.0f;
    const float progress_sigma_m = 250.0f;
    const float distance_weight = 0.001f;
    const float distance_scale_m = 532.67f;
    const float rate_weight = 2e-4f;
    const float fail_penalty = 30.0f;

    float previous_distance = state->values[PREVIOUS_DISTANCE_SLOT];
    float previous_potential = expf(-previous_distance / progress_sigma_m);
    float potential = expf(-step->distance_m / progress_sigma_m);
    float reward =
        progress_weight * (potential - previous_potential) -
        distance_weight * (step->distance_m / distance_scale_m) -
        rate_weight * step->rate_norm -
        (step->failed ? fail_penalty : 0.0f);
    state->values[PREVIOUS_DISTANCE_SLOT] = step->distance_m;
    return isfinite(reward) ? reward : 0.0f;
}
