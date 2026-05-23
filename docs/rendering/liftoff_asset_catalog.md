# Liftoff Asset Catalog For Rendering

This document catalogs the local Liftoff install enough to start replacing the
placeholder target visual with a drone model. The raw Liftoff assets are not
checked into this repository. Treat the Steam install as a local input and keep
any extracted meshes/textures in a user-local cache unless licensing is
explicitly settled.

## Local Install

- Steam app: Liftoff, app id `410340`
- Steam manifest: `/mnt/c/Program Files (x86)/Steam/steamapps/appmanifest_410340.acf`
- Windows manifest path: `C:\Program Files (x86)\Steam\steamapps\appmanifest_410340.acf`
- WSL install path: `/mnt/c/Program Files (x86)/Steam/steamapps/common/Liftoff`
- Windows install path: `C:\Program Files (x86)\Steam\steamapps\common\Liftoff`
- Unity data path: `C:\Program Files (x86)\Steam\steamapps\common\Liftoff\Liftoff_Data`
- Local user data: `C:\Users\jpeng\AppData\LocalLow\LuGus Studios\Liftoff`
- Observed build id: `22928800`

The current metadata scan is stored outside version control at
`.runs/liftoff_asset_catalog.json`.

## Main Containers

The Unity data directory contains `resources.assets`, `sharedassets*.assets`,
`level*`, and external payload files such as `.resS` and `.resource`.

Large payloads that likely hold mesh and texture data:

- `sharedassets3.assets.resS`: about 511 MB
- `sharedassets6.assets.resS`: about 187 MB
- `sharedassets15.assets.resS`: about 152 MB
- `resources.assets.resS`: about 113 MB
- `sharedassets5.assets.resS`: about 89 MB
- `level3.resS`: about 70 MB

## Drone Model Candidates

Start with the Vortex setup in `resources.assets`; it has a coherent frame,
arms, battery, props, camera mount, and a default drone configuration.

### Frame And Body

Container: `resources.assets`

- Material `VortexFrame01`, path id `7`
- Material `VortexMisc01`, path id `8`
- Material `VortexBattery02`, path id `13`
- Material `VortexBattery02Strap01`, path id `14`
- Mesh `VortexFrame01`, path id `305`
- Mesh `VortexArmLB01`, path id `312`
- Mesh `VortexArmRB01`, path id `319`
- Mesh `VortexArmRF01`, path id `323`
- Mesh `VortexArmLF01`, path id `327`
- Mesh `VortexBattery01`, path id `302`
- Mesh `VortexBattery01Strap03`, path id `310`
- Mesh `VortexBattery01Strap04`, path id `314`
- Textures `VortexFrame01`, `VortexFrame01_M`, `VortexFrame01_E`, path ids `297`, `255`, `238`
- Texture `VortexMisc01`, path id `295`
- Texture `VortexMisc01_M`, path id `216`
- Textures `VortexBattery02`, `VortexBattery02_M`, path ids `234`, `94`
- GameObjects `VortexFrame01` and `FrameVortex01`, path ids `651`, `702`
- GameObjects `MotorRFSlot`, `MotorRBSlot`, `MotorLFSlot`, `MotorLBSlot`, path ids `649`, `668`, `716`, `663`

### Propellers

Container: `resources.assets`

- Material `Dal5050TriPropeller01`, path id `16`
- Mesh `Dal5050Propeller3R01`, path id `301`
- Mesh `Dal5050Propeller3L01`, path id `309`
- Mesh `GemfanBullnose5045Propeller2L01`, path id `308`
- Textures `PropBasePlastic01`, `PropBasePlastic01_M`, `PropBasePlastic01_mask`, path ids `284`, `155`, `158`
- Texture `HQProp603501_N`, path id `136`
- GameObjects `Dal5050TriPropeller01_L(Clone)` and `Dal5050TriPropeller01_R(Clone)`, path ids `633`, `693`, `673`, `703`
- GameObjects `Prop01`, `Prop02`, `Prop03`, `Prop04`, path ids `647`, `718`, `661`, `643`

Container: `sharedassets5.assets`

- Material `Racekraft5051Propeller01`, path id `9`
- Mesh `RaceKraft5051Propeller3R02`, path id `41`
- Mesh `RaceKraft5051Propeller3L02`, path id `45`

Container: `sharedassets15.assets`

- Material `HQV1Series5050TriPropeller01`, path id `10`
- GameObjects `RaceKraft5051Propeller3R01_R`, `RaceKraft5051Propeller3L01_L`, and copies, path ids `219`, `220`, `183`, `131`
- GameObject `HQV1Series5040TriPropeller01`, path id `115`

### Motors

Container: `sharedassets5.assets`

- Material `XNova2205Motor01`, path id `8`
- Mesh `XnovaMotor2205_Up`, path id `37`
- Mesh `XnovaMotor2205_Low`, path id `43`
- Textures `XnovaMotor2205-2600KV_01`, `XnovaMotor2205_M`, `XnovaMotor2205_N`, path ids `15`, `28`, `33`
- PhysicMaterial `DronePhysicsMaterial01`, path id `51`

Container: `sharedassets15.assets`

- Material `Hyperlite22062140Motor01`, path id `8`
- Material `HypetrainStingersswarm2207Motor01`, path id `9`
- Textures `HyperLite2206KV2140Motor01`, `HyperLite2206KV2140Motor01_M`, `HyperLite2206KV2140Motor01_N`, path ids `43`, `21`, `36`
- Textures `HypetrainStingersswarm2207Motor01`, `HypetrainStingersswarm2207Motor01_M`, `HypetrainStingersswarm2207Motor01_N`, path ids `28`, `15`, `31`
- GameObjects for `Hyperlite22062140Motor01` and `HypetrainStingersswarm2207Motor01`, path ids `181`, `204`, `208`, `214`, `121`, `167`, `200`, `237`

### FPV Camera And Mounts

Container: `resources.assets`

- Material `RedotCamMini01`, path id `15`
- Textures `RedotMini01`, `RedotMini01_M`, `RedotMini01_N`, path ids `140`, `117`, `188`
- GameObjects `CameraSlot`, `CameraHook`, `CameraParentPoint`, `CameraRedotMini01`, path ids `681`, `734`, `819`, `1036`

Container: `sharedassets5.assets`

- Material `RedotCam01`, path id `5`
- Textures `RedotCam01`, `RedotCam01_M`, `RedotCam01_N`, path ids `31`, `23`, `21`

### Configuration, Animation, And Audio

Container: `resources.assets`

- TextAsset `DefaultDroneConfiguration01`, path id `456`
- AudioClip `DroneFlyLoop01`, path id `486`
- AudioClip `DroneFlyLoop01_`, path id `487`
- Texture `LOSDroneLocator01`, path id `242`

Container: `sharedassets6.assets`

- AnimationClips `GettingStartedDrone@Startup01`, `GettingStartedDrone@Hover01`, `GettingStartedDrone@FlyAway01`, path ids `83`, `82`, `81`
- AnimatorController `GettingStartedDrone`, path id `91`

Container: `sharedassets15.assets`

- GameObjects `DroneVisualizer`, `DroneDummy01`, `DroneBoundingBox`, path ids `298`, `478`, `487`

## Environment And FPV Context

Environment work is secondary for now, but the scan found useful starting
points:

- `sharedassets0.assets`: `Default-Skybox`, plus FPV brand/logo textures
- `sharedassets3.assets`: `SkyBlueCloudy01*` skybox textures, terrain textures such as `TerrainWetMud01`, `TerrainForestNeedles01`, `TerrainWoodChipsGrass01`, and `TerrainAsphaltPlants01`
- `sharedassets5.assets`: `ParisDroneFestival` cubemap, `DroneSharingFloor01`, `DroneSharingSkybox01`
- `sharedassets6.assets`: `LensDirt01`, FPV goggles strap material/textures, `GoggleStrap`
- `sharedassets13.assets`: `TestBardwellSkybox`
- `sharedassets14.assets`: track editor checkpoint/environment UI objects
- `resources.assets`: tutorial track TextAssets `09_FinalBasicTrack`, `10_FinalAdvancedTrack`, `TrackEditorGizmo`, `TrackSnapAttachment01`, `TrackSnapPivot01`, environment and race dropdown/panel objects
- `sharedassets8.assets`: menu thumbnails for track builder and race modes

## Rendering Plan

1. Add a dev-only asset catalog/export command.

   The command should read the local Liftoff install path, resolve the container
   and path ids above, and write metadata plus derived runtime artifacts under a
   cache directory such as `.runs/liftoff_assets/`. Do not commit extracted
   Liftoff meshes or textures.

   Current command:

   ```bash
   python -m rendering.python.liftoff_assets --out-dir .runs/liftoff_assets --all-variants
   ```

   Current variants:

   - `vortex_dal_xnova_runcam`: DAL tri-blades, XNova motors, compact Runcam-style camera
   - `vortex_racekraft_xnova_hs1177`: RaceKraft tri-blades, XNova motors, HS1177 box camera
   - `vortex_gemfan_xnova_actioncam`: Gemfan bullnose props, XNova motors, tall action camera
   - `vortex_dal_heavy_actioncam`: DAL props, oversized motor bells, top-mounted action camera
   - `vortex_racekraft_low_cam`: RaceKraft props, smaller motor bells, low forward camera mount

2. Export the first target drone asset set.

   Use `DefaultDroneConfiguration01` to confirm the intended part combination.
   If that config is hard to decode, start with a fixed assembly:
   `VortexFrame01`, four Vortex arms, `XnovaMotor2205` upper/lower motor
   meshes, left/right DAL or RaceKraft prop meshes, battery, straps, and Redot
   camera. Export into a simple repo-owned format, preferably glTF for
   inspection plus a compact binary format for the native renderer if needed.

3. Add mesh support to the native renderer.

   The current renderer is a C++ software renderer. Add repo-owned mesh
   structures for vertices, normals, UVs, indices, material ids, texture ids,
   and object transforms. Implement a z-buffered triangle path first with flat
   or normal-based shading. Texture sampling can follow once geometry alignment
   is correct.

4. Assemble the drone prefab in renderer-owned code.

   Build the drone from part transforms instead of baking one monolithic mesh.
   Use the existing target position from the sim for placement. Keep the current
   velocity-derived yaw/roll approximation until the sim/render ABI exposes a
   target attitude quaternion.

5. Replace the software silhouette behind a feature gate.

   Keep the current procedural quadcopter as a fallback when Liftoff assets are
   unavailable. Add a config flag or asset availability check so tests can run
   without depending on the local Steam install.

6. Add visual effects that matter for FPV readability.

   Add prop blur disks or translucent swept blades, preserve the existing
   lens/vignette/noise path, and optionally use `LensDirt01` as an overlay.
   Match Liftoff's drone readability from the pursuer POV before spending time
   on full environment import.

7. Defer full environments.

   Catalogue skyboxes, terrain textures, and track objects now, but implement
   them after the drone target is real. The immediate goal is a recognizable
   Liftoff-style drone target in the pursuer POV, not full scene parity.

## Open Technical Questions

- Whether `DefaultDroneConfiguration01` encodes the exact default parts and
  transform offsets directly enough to drive prefab assembly.
- Whether Unity mesh extraction produces usable local coordinates for the frame,
  arms, motors, and prop slots without also walking prefab transforms.
- Whether we should load glTF directly in C++ or use a small preconverted binary
  format for deterministic tests and simpler native code.
- How soon the sim ABI should expose target attitude. Without attitude, the
  renderer can only infer orientation from velocity.
