"""Noise / covariance defaults for paper_sim.

Paper §II-B specifies the noise *structure* (zero-mean Gaussian for IMU and
image, Wiener-process biases) but does not publish numeric values for any of
σ_gyr, σ_acc, σ_b_gyr, σ_b_acc, σ_img, Q, R, P_0. Defaults below are typical
Pixhawk-class MEMS IMU + 1 px/centroid camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NoiseConfig:
    # ── IMU white noise (Eqs. 7, 9) ───────────────────────────────────
    sigma_gyr: float = 0.01      # rad/s
    sigma_acc: float = 0.05      # m/s^2

    # ── IMU bias random walk (Eqs. 8, 10) ─────────────────────────────
    sigma_b_gyr: float = 1.0e-4  # rad/s/√s
    sigma_b_acc: float = 1.0e-3  # m/s²/√s
    bias_init_std: float = 0.005

    # ── Image measurement noise (Eq. 11) ──────────────────────────────
    # Normalized image-coord std. Approx σ_pixel_px / f_oc.
    sigma_img: float = 1.0e-3

    # ── DKF process noise Q (per √dt, applied to predicted covariance) ─
    Q_q: float = 1.0e-6
    Q_pr: float = 1.0e-4
    Q_vr: float = 1.0e-3
    Q_ip: float = 1.0e-6
    Q_b_gyr: float = 1.0e-8
    Q_b_acc: float = 1.0e-7

    # ── DKF initial covariance P_0 (diagonal blocks) ──────────────────
    P0_q: float = 1.0e-3
    P0_pr: float = 0.5
    P0_vr: float = 0.25
    P0_ip: float = 1.0e-2
    P0_b_gyr: float = 1.0e-4
    P0_b_acc: float = 1.0e-3

    rng_seed: int = 0
