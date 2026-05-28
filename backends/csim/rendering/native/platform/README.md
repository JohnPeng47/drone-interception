# Platform Boundary

Platform code is boxed behind `render_platform.h`. The rest of the renderer
must not include Windows or Linux system headers directly.

## Windows

`platform/win32` will own:

- named pipe or shared-memory handle creation
- Win32 events/semaphores for frame readiness
- handle inheritance/security descriptors
- Windows path and process-discovery details for Unity
- timeout conversion to Win32 wait APIs

No Windows types should appear in `backends/csim/rendering/include` or `backends/csim`.

## Linux

`platform/linux` will own:

- POSIX shared memory
- `eventfd`, `poll`, or futex-backed readiness
- Linux Unity runtime process discovery if needed
- timeout conversion to POSIX wait APIs

The public C API stays the same when switching platforms.
