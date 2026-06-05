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
    const float catch_reward = 10.0f;
    const float distance_weight = 0.001f;
    const float progress_weight = 0.1f;
    const float fail_penalty = 30.0f;
    const float rate_weight = 2e-4f;

    float previous_distance = state->values[PREVIOUS_DISTANCE_SLOT];
    float progress = previous_distance - step->distance_m;
    float reward =
        (step->intercepted ? catch_reward : 0.0f) +
        progress_weight * progress -
        distance_weight * step->distance_m -
        rate_weight * step->rate_norm -
        (step->failed ? fail_penalty : 0.0f);
    state->values[PREVIOUS_DISTANCE_SLOT] = step->distance_m;
    return isfinite(reward) ? reward : 0.0f;
}
