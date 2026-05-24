# Liftoff Render Bridge

This is the Liftoff-side half of `SimConfig.render_frames`.

The bridge is intended to run inside the Liftoff Unity process through a Mono
loader such as BepInEx 5 x64. It receives one selected SimEngine drone/camera
state over loopback TCP, drives Liftoff's existing first-person camera stack,
captures the rendered frame, and returns raw RGB bytes.

The source is intentionally kept out of the Steam install. Build/install should
copy the compiled plugin into the loader's plugin folder only after the loader
path is explicit.

Relevant Liftoff classes found during recon:

- `FlightCameraControllerDefault`
- `TwinCamera`
- `LiftoffFisheye`
- `CameraNoise`

The bridge should bind `TwinCamera.PrimaryCam` and `TwinCamera.SetFOV` so the
real Liftoff FOV/fisheye path remains in control.
