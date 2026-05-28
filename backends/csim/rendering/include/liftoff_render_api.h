#pragma once

#include "liftoff_render_errors.h"
#include "liftoff_render_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LiftoffRenderEngine LiftoffRenderEngine;

LiftoffRenderStatus liftoff_render_engine_create(
    const LiftoffRenderConfig* config,
    LiftoffRenderEngine** engine
);

void liftoff_render_engine_destroy(LiftoffRenderEngine* engine);

LiftoffRenderStatus liftoff_render_frame(
    LiftoffRenderEngine* engine,
    const LiftoffRenderFrameRequest* request,
    LiftoffRenderFrame* frame
);

void liftoff_render_release_frame(
    LiftoffRenderEngine* engine,
    LiftoffRenderFrame* frame
);

#ifdef __cplusplus
}
#endif
