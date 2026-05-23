#include "render_platform.h"

#ifdef _WIN32

namespace liftoff_render {
namespace {

class Win32PlatformTransport final : public PlatformTransport {
public:
    LiftoffRenderStatus open(const PlatformConfig& config) override {
        (void)config;
        return LIFTOFF_RENDER_BACKEND_UNAVAILABLE;
    }

    LiftoffRenderStatus transact(ByteView request, std::vector<uint8_t>* response) override {
        (void)request;
        (void)response;
        return LIFTOFF_RENDER_BACKEND_UNAVAILABLE;
    }

    void close() override {}
};

}  // namespace

std::unique_ptr<PlatformTransport> create_platform_transport(
    LiftoffRenderPlatformKind platform
) {
    if (platform == LIFTOFF_RENDER_PLATFORM_AUTO ||
            platform == LIFTOFF_RENDER_PLATFORM_WINDOWS) {
        return std::unique_ptr<PlatformTransport>(new Win32PlatformTransport());
    }
    return nullptr;
}

}  // namespace liftoff_render

#endif
