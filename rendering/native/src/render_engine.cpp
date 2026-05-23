#include "liftoff_render_api.h"

#include <cstring>
#include <memory>

struct LiftoffRenderEngine {
    LiftoffRenderConfig config;
};

const char* liftoff_render_status_string(LiftoffRenderStatus status) {
    switch (status) {
        case LIFTOFF_RENDER_OK:
            return "ok";
        case LIFTOFF_RENDER_DISABLED:
            return "rendering disabled";
        case LIFTOFF_RENDER_BACKEND_UNAVAILABLE:
            return "render backend unavailable";
        case LIFTOFF_RENDER_TIMEOUT:
            return "render timed out";
        case LIFTOFF_RENDER_INVALID_REQUEST:
            return "invalid render request";
        case LIFTOFF_RENDER_FRAME_DROPPED:
            return "render frame dropped";
        case LIFTOFF_RENDER_INTERNAL_ERROR:
            return "internal render error";
        default:
            return "unknown render status";
    }
}

LiftoffRenderStatus liftoff_render_engine_create(
    const LiftoffRenderConfig* config,
    LiftoffRenderEngine** engine
) {
    if (engine == nullptr) {
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }
    *engine = nullptr;
    if (config == nullptr) {
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }

    std::unique_ptr<LiftoffRenderEngine> created(new LiftoffRenderEngine{});
    std::memcpy(&created->config, config, sizeof(LiftoffRenderConfig));
    *engine = created.release();
    return LIFTOFF_RENDER_OK;
}

void liftoff_render_engine_destroy(LiftoffRenderEngine* engine) {
    delete engine;
}

LiftoffRenderStatus liftoff_render_frame(
    LiftoffRenderEngine* engine,
    const LiftoffRenderFrameRequest* request,
    LiftoffRenderFrame* frame
) {
    if (frame != nullptr) {
        std::memset(frame, 0, sizeof(LiftoffRenderFrame));
    }
    if (engine == nullptr || request == nullptr || request->drone == nullptr ||
            request->camera == nullptr || frame == nullptr) {
        return LIFTOFF_RENDER_INVALID_REQUEST;
    }
    if (engine->config.backend == LIFTOFF_RENDER_BACKEND_NONE) {
        return LIFTOFF_RENDER_DISABLED;
    }

    return LIFTOFF_RENDER_BACKEND_UNAVAILABLE;
}

void liftoff_render_release_frame(
    LiftoffRenderEngine* engine,
    LiftoffRenderFrame* frame
) {
    (void)engine;
    if (frame != nullptr) {
        std::memset(frame, 0, sizeof(LiftoffRenderFrame));
    }
}
