#include "intercept_native.c"
#include "reward_api.h"

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float catches;
    float failures;
    float timeouts;
    float n;
} Log;

typedef struct {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;
    NativeScenario* scenarios;
    int scenario_count;
    int cursor;
    int scenario_index;
    SimEngine engine;
    float elapsed_s;
    int episode_length;
    float episode_return;
    NativeRewardState reward_state;
    int* shared_cursor;
    int max_episode_steps;
} Env;

void c_reset(Env* env);
void c_step(Env* env);
void c_close(Env* env);
void c_render(Env* env);

#define OBS_SIZE 25
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor
#define Env Env

#include "vecenv.h"

typedef struct {
    StaticVec* vec;
    NativeScenario* scenarios;
    int scenario_count;
    int cursor;
    int max_episode_steps;
} PufferInterceptVec;

typedef int cudaError_t;
typedef int cudaMemcpyKind;
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) { (void)flags; *ptr = malloc(size); return *ptr == NULL; }
cudaError_t cudaMalloc(void** ptr, size_t size) { *ptr = malloc(size); return *ptr == NULL; }
cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) { (void)kind; memcpy(dst, src, size); return 0; }
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) { (void)stream; return cudaMemcpy(dst, src, size, kind); }
cudaError_t cudaMemset(void* ptr, int value, size_t size) { memset(ptr, value, size); return 0; }
cudaError_t cudaFree(void* ptr) { free(ptr); return 0; }
cudaError_t cudaFreeHost(void* ptr) { free(ptr); return 0; }
cudaError_t cudaSetDevice(int device) { (void)device; return 0; }
cudaError_t cudaDeviceSynchronize(void) { return 0; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; return 0; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) { (void)flags; *stream = NULL; return 0; }
cudaError_t cudaStreamQuery(cudaStream_t stream) { (void)stream; return 0; }
const char* cudaGetErrorString(cudaError_t err) { (void)err; return "cpu stub"; }

static void reset_env_slot(Env* env) {
    if (env == NULL || env->scenario_count <= 0) return;
    int scenario_index = 0;
    if (env->shared_cursor != NULL) {
        scenario_index = (*env->shared_cursor)++;
    } else {
        scenario_index = env->cursor++;
    }
    env->scenario_index = scenario_index % env->scenario_count;
    NativeScenario* scenario = &env->scenarios[env->scenario_index];
    sim_engine_init(&env->engine, scenario->pursuer_params, scenario->pursuer_initial);
    sim_engine_set_intercept_radius(&env->engine, scenario->intercept_radius_m);
    sim_engine_set_targets(&env->engine, scenario->targets, scenario->num_targets);
    sim_engine_set_cameras(&env->engine, scenario->cameras, scenario->num_cameras);
    env->elapsed_s = 0.0f;
    env->episode_length = 0;
    env->episode_return = 0.0f;
    native_reward_reset(&env->reward_state, &env->engine, scenario);
    write_observation(&env->engine, NULL, env->observations);
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->scenarios = (NativeScenario*)dict_get(kwargs, "scenarios")->ptr;
    env->scenario_count = (int)dict_get(kwargs, "scenario_count")->value;
    env->shared_cursor = (int*)dict_get(kwargs, "cursor")->ptr;
    env->max_episode_steps = (int)dict_get(kwargs, "max_episode_steps")->value;
    env->cursor = 0;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "catches", log->catches);
    dict_set(out, "failures", log->failures);
    dict_set(out, "timeouts", log->timeouts);
}

void c_reset(Env* env) {
    reset_env_slot(env);
}

void c_step(Env* env) {
    NativeScenario* scenario = &env->scenarios[env->scenario_index];
    float previous_action[4] = {
        env->actions[0],
        env->actions[1],
        env->actions[2],
        env->actions[3],
    };
    float max_thrust = scenario->max_thrust_n > 0.0f
        ? scenario->max_thrust_n
        : scenario->pursuer_params.mass * scenario->pursuer_params.gravity * 2.0f;
    float max_rate = scenario->max_rate_rps > 0.0f ? scenario->max_rate_rps : scenario->pursuer_params.max_omega;
    float thrust = clampf((env->actions[0] + 1.0f) * 0.5f, 0.0f, 1.0f) * max_thrust;
    float rates[3] = {
        clampf(env->actions[1], -1.0f, 1.0f) * max_rate,
        clampf(env->actions[2], -1.0f, 1.0f) * max_rate,
        clampf(env->actions[3], -1.0f, 1.0f) * max_rate,
    };

    sim_engine_step_ctbr_dt(
        &env->engine,
        thrust,
        rates,
        max_thrust,
        max_rate,
        scenario->rpm_min,
        scenario->k_w,
        scenario->dt,
        scenario->substeps
    );
    env->elapsed_s += scenario->dt;
    env->episode_length += 1;

    int intercepted = env->engine.metrics.intercepted ? 1 : 0;
    int failed = 0;
    const State* state = &env->engine.pursuer.state;
    if (!isfinite(state->pos.x) || !isfinite(state->pos.y) || !isfinite(state->pos.z)) {
        failed = 1;
    } else if (scenario->has_bounds) {
        failed = fabsf(state->pos.x) > scenario->bounds_w.x ||
            fabsf(state->pos.y) > scenario->bounds_w.y ||
            fabsf(state->pos.z) > scenario->bounds_w.z;
    }
    int timeout = (scenario->duration_s > 0.0f && env->elapsed_s >= scenario->duration_s) ||
        (env->max_episode_steps > 0 && env->episode_length >= env->max_episode_steps);
    float rate_norm = sqrtf(rates[0] * rates[0] + rates[1] * rates[1] + rates[2] * rates[2]);
    float distance = env->engine.metrics.distance_m;
    NativeRewardStep reward_step = {
        .engine = &env->engine,
        .scenario = scenario,
        .action = env->actions,
        .previous_action = previous_action,
        .body_rates_b = rates,
        .thrust_n = thrust,
        .max_thrust_n = max_thrust,
        .max_rate_rps = max_rate,
        .previous_distance_m = env->reward_state.values[0],
        .distance_m = distance,
        .rate_norm = rate_norm,
        .elapsed_s = env->elapsed_s,
        .episode_length = env->episode_length,
        .intercepted = intercepted,
        .failed = failed,
        .timeout = timeout,
    };
    float reward = native_reward_step(&env->reward_state, &reward_step);
    env->episode_return += reward;
    env->rewards[0] = reward;
    env->terminals[0] = (float)(intercepted || failed || timeout);

    if (intercepted || failed || timeout) {
        env->log.episode_return = env->episode_return;
        env->log.episode_length = (float)env->episode_length;
        env->log.catches += intercepted ? 1.0f : 0.0f;
        env->log.failures += failed ? 1.0f : 0.0f;
        env->log.timeouts += timeout ? 1.0f : 0.0f;
        env->log.n += 1.0f;
        return;
    }

    write_observation(&env->engine, previous_action, env->observations);
}

void c_render(Env* env) {
    (void)env;
}

void c_close(Env* env) {
    (void)env;
}

int puffer_intercept_create(const char* scenario_path, int total_agents, int num_buffers, int max_episode_steps, PufferInterceptVec** out) {
    if (scenario_path == NULL || total_agents <= 0 || num_buffers <= 0 || out == NULL) return 0;
    PufferInterceptVec* handle = (PufferInterceptVec*)calloc(1, sizeof(PufferInterceptVec));
    if (handle == NULL) return 0;
    if (!load_scenarios(scenario_path, &handle->scenarios, &handle->scenario_count)) {
        free(handle);
        return 0;
    }
    Dict* vec_kwargs = create_dict(2);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);
    Dict* env_kwargs = create_dict(4);
    handle->cursor = 0;
    handle->max_episode_steps = max_episode_steps;
    env_kwargs->items[0] = (DictItem){.key = "scenarios", .value = 0.0, .ptr = handle->scenarios};
    env_kwargs->items[1] = (DictItem){.key = "scenario_count", .value = (double)handle->scenario_count, .ptr = NULL};
    env_kwargs->items[2] = (DictItem){.key = "cursor", .value = 0.0, .ptr = &handle->cursor};
    env_kwargs->items[3] = (DictItem){.key = "max_episode_steps", .value = (double)handle->max_episode_steps, .ptr = NULL};
    env_kwargs->size = 4;
    handle->vec = create_static_vec(total_agents, num_buffers, 0, vec_kwargs, env_kwargs);
    free(vec_kwargs->items);
    free(vec_kwargs);
    free(env_kwargs->items);
    free(env_kwargs);
    if (handle->vec == NULL) {
        free(handle->scenarios);
        free(handle);
        return 0;
    }
    *out = handle;
    return 1;
}

void puffer_intercept_destroy(PufferInterceptVec* handle) {
    if (handle == NULL) return;
    if (handle->vec != NULL) static_vec_close(handle->vec);
    free(handle->scenarios);
    free(handle);
}

void puffer_intercept_reset(PufferInterceptVec* handle) {
    if (handle == NULL || handle->vec == NULL) return;
    handle->cursor = 0;
    static_vec_reset(handle->vec);
}

void puffer_intercept_step(PufferInterceptVec* handle) {
    if (handle == NULL || handle->vec == NULL) return;
    cpu_vec_step(handle->vec);
    Env* envs = (Env*)handle->vec->envs;
    for (int i = 0; i < handle->vec->size; i++) {
        if (envs[i].terminals[0] > 0.5f) {
            reset_env_slot(&envs[i]);
        }
    }
}

float* puffer_intercept_observations(PufferInterceptVec* handle) {
    return handle == NULL || handle->vec == NULL ? NULL : (float*)handle->vec->observations;
}

float* puffer_intercept_actions(PufferInterceptVec* handle) {
    return handle == NULL || handle->vec == NULL ? NULL : handle->vec->actions;
}

float* puffer_intercept_rewards(PufferInterceptVec* handle) {
    return handle == NULL || handle->vec == NULL ? NULL : handle->vec->rewards;
}

float* puffer_intercept_terminals(PufferInterceptVec* handle) {
    return handle == NULL || handle->vec == NULL ? NULL : handle->vec->terminals;
}

int puffer_intercept_scenario_count(PufferInterceptVec* handle) {
    return handle == NULL ? 0 : handle->scenario_count;
}
