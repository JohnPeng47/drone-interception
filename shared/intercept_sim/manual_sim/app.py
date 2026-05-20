from __future__ import annotations

from pathlib import Path

import numpy as np

from intercept_sim.manual_sim.config import load_manual_sim_config
from intercept_sim.manual_sim.control_map import ControlMap
from intercept_sim.manual_sim.renderer import BasicOpenGlRenderer, RenderFrame, RendererConfig
from intercept_sim.rotorpy_adapter import ctbr_to_rotorpy


def run_manual_sim(config_path: str | Path) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Manual sim requires pygame. Install intercept-sim[manual].") from exc

    from rotorpy.vehicles.hummingbird_params import quad_params
    from rotorpy.vehicles.multirotor import Multirotor

    config = load_manual_sim_config(config_path)
    initial_state = _initial_state(config.raw, quad_params)
    vehicle = Multirotor(
        quad_params,
        initial_state=initial_state,
        control_abstraction="cmd_ctbr",
        aero=bool(config.vehicle.get("aero", True)),
        enable_ground=bool(config.vehicle.get("enable_ground", True)),
    )
    control_map = ControlMap.from_config(config.control, mass_kg=quad_params["mass"])
    renderer = BasicOpenGlRenderer(
        RendererConfig(
            ground_size_m=float(config.raw["world"]["ground"]["size_m"]),
            arm_length_m=float(config.renderer["drone_model"]["arm_length_m"]),
        )
    )
    renderer.initialize()

    state = initial_state
    clock = pygame.time.Clock()
    running = True
    t = 0.0
    dt = config.sim_dt
    while running:
        pressed_names = _pressed_key_names(pygame)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if control_map.should_quit(pressed_names):
            running = False

        command = control_map.update(pressed_names, dt, t=t)
        state = vehicle.step(state, ctbr_to_rotorpy(command), dt)
        renderer.render(RenderFrame.from_rotorpy_state(t, state))
        t += dt
        clock.tick(config.render_hz)

    pygame.quit()


def _pressed_key_names(pygame_module: object) -> set[str]:
    pressed = pygame_module.key.get_pressed()
    names: set[str] = set()
    key_constants = {
        "W": pygame_module.K_w,
        "A": pygame_module.K_a,
        "S": pygame_module.K_s,
        "D": pygame_module.K_d,
        "UP": pygame_module.K_UP,
        "DOWN": pygame_module.K_DOWN,
        "LEFT": pygame_module.K_LEFT,
        "RIGHT": pygame_module.K_RIGHT,
        "SPACE": pygame_module.K_SPACE,
        "R": pygame_module.K_r,
        "ESCAPE": pygame_module.K_ESCAPE,
    }
    for name, code in key_constants.items():
        if pressed[code]:
            names.add(name)
    return names


def _initial_state(raw_config: dict[str, object], quad_params: dict[str, object]) -> dict[str, np.ndarray]:
    vehicle = raw_config["vehicle"]
    assert isinstance(vehicle, dict)
    raw_state = vehicle["initial_state"]
    assert isinstance(raw_state, dict)
    hover_speed = np.sqrt(float(quad_params["mass"]) * 9.81 / (float(quad_params["num_rotors"]) * float(quad_params["k_eta"])))
    return {
        "x": np.asarray(raw_state["position_w_m"], dtype=float),
        "v": np.asarray(raw_state["velocity_w_mps"], dtype=float),
        "q": np.asarray(raw_state["quat_wb_xyzw"], dtype=float),
        "w": np.asarray(raw_state["body_rates_b_rps"], dtype=float),
        "wind": np.zeros(3, dtype=float),
        "rotor_speeds": np.full(int(quad_params["num_rotors"]), hover_speed, dtype=float),
    }
