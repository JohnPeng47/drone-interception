#pragma once

#include "liftoff_render_errors.h"
#include "liftoff_render_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace liftoff_render {

struct PlatformConfig {
    LiftoffRenderPlatformKind platform;
    uint32_t timeout_ms;
    const char* scene_id;
};

struct ByteView {
    const uint8_t* data;
    size_t size;
};

class PlatformTransport {
public:
    virtual ~PlatformTransport() = default;
    virtual LiftoffRenderStatus open(const PlatformConfig& config) = 0;
    virtual LiftoffRenderStatus transact(ByteView request, std::vector<uint8_t>* response) = 0;
    virtual void close() = 0;
};

std::unique_ptr<PlatformTransport> create_platform_transport(
    LiftoffRenderPlatformKind platform
);

}  // namespace liftoff_render
