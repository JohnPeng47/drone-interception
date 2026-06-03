# Beihang M4 T2 caught-region analysis

Generated from trial CSVs under `.agents/heuristic-rl-group/beihang-m4-t2-20260601/controller-001-beihang_minimal_sim` joined to records from `scripts/generators/robust_intercept.py`.

## Main result

The caught cases form a broad lobe, not a single tight point cluster. `camera_azimuth_deg` is not informative for catch geometry because the current robust-intercept setup is yaw-symmetric about gravity; the useful reduced coordinates are `camera_elevation_deg`, `camera_u_fraction`, and `camera_v_fraction`, plus the range/speed grid.

For `final-validation-512`, catches are concentrated in near-range and level-to-descending geometries:
- Total caught: 32 / 512 (6.25%).
- Range rates: R=5m: 19/171, R=8m: 8/171, R=20m: 5/170.
- The strongest elevation band is `camera_elevation_deg` from -15 to 0 deg, with adjacent bands -30 to -15 deg and -45 to -30 deg also elevated.
- The strongest horizontal image-plane band is near center/slightly right: `camera_u_fraction` in [0, 0.3), followed by [-0.3, 0).
- `camera_v_fraction` is not a clean separator; caught cases exist across most vertical FOV values.
- A useful approximate high-yield region is `range <= 8 m`, `camera_elevation_deg` in [-45, 15], and `camera_u_fraction` in [-0.3, 0.3]. It contains 18 of the 32 caught validation samples while selecting 52 of 512 total samples.

For the best 128-sample milestone (`milestone-011-stronger-aligned-thrust`):
- Total caught: 13 / 128 (10.16%).
- It shows the same broad lobe, but the coarser 128 table makes individual caught seeds look more scattered.

## Artifacts

- `joined_trials.csv`: every trial row joined with robust-intercept sample coordinates.
- `caught_seed_summary.csv`: one row per source-table seed that was caught at least once.
- `region_summary.json`: machine-readable binned rates and feature ranges.
- `final_validation_512_projections.png` and `milestone_011_best_128_projections.png`: visual projections of the reduced sample space.
- `sobol_samples_512_caught_union_3d.html`: interactive 3D scatter of all 512 sampled points, with every caught seed colored orange.
- `sobol_samples_128_caught_union_3d.html`: equivalent interactive 3D scatter for the 128-sample milestone table.
- `final_validation_512_caught_3d.html`: 3D scatter colored by catches in only the final validation run.
- `sobol_samples_512_target_relative_r8_3d.html`: interactive 3D target-relative view with the pursuer at the origin and target endpoints normalized to 8 m.
- `sobol_samples_128_target_relative_r8_3d.html`: equivalent target-relative view for the 128-sample table.
- `final_validation_512_target_relative_r8_3d.html`: target-relative view colored by catches in only the final validation run.
- `sobol_samples_512_target_relative_r8_camera_elevation_groups_3d.html`: target-relative 3D view with caught rays grouped by camera-elevation band.
- `sobol_samples_128_target_relative_r8_camera_elevation_groups_3d.html`: equivalent grouped view for the 128-sample table.
- `final_validation_512_target_relative_r8_camera_elevation_groups_3d.html`: final-validation grouped view.
