#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum LiftoffRenderStatus {
    LIFTOFF_RENDER_OK = 0,
    LIFTOFF_RENDER_DISABLED = 1,
    LIFTOFF_RENDER_BACKEND_UNAVAILABLE = 2,
    LIFTOFF_RENDER_TIMEOUT = 3,
    LIFTOFF_RENDER_INVALID_REQUEST = 4,
    LIFTOFF_RENDER_FRAME_DROPPED = 5,
    LIFTOFF_RENDER_INTERNAL_ERROR = 6,
} LiftoffRenderStatus;

const char* liftoff_render_status_string(LiftoffRenderStatus status);

#ifdef __cplusplus
}
#endif
