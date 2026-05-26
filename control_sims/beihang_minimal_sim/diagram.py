"""Build the minimal Drake interception diagram."""

from __future__ import annotations

from pydrake.systems.framework import Diagram, DiagramBuilder

from .config import TrialConfig, initial_rotation_wb_toward_target
from .controller.beihang_baseline_strategy import BeihangBaselineStrategy
from .controller.heuristic_strategy_system import HeuristicStrategySystem
from .scoring.capture_status_system import CaptureStatusSystem
from .scoring.trial_logger import TrialLogger
from .sensing.image_feature_system import ImageFeatureSystem
from .sensing.pinhole_camera_system import PinholeCameraSystem
from .world.point_mass_ctbr_plant import PointMassCtbrPlant
from .world.scene_assembler import SceneAssembler
from .world.target_motion_system import TargetMotionSystem


def build_minimal_diagram(config: TrialConfig) -> tuple[Diagram, TrialLogger]:
    builder = DiagramBuilder()
    plant = builder.AddSystem(
        PointMassCtbrPlant(
            config=config.vehicle,
            initial_rotation_wb=initial_rotation_wb_toward_target(config),
            dt=config.dt,
        )
    )
    target = builder.AddSystem(TargetMotionSystem(config.target))
    scene = builder.AddSystem(SceneAssembler())
    camera = builder.AddSystem(PinholeCameraSystem(config.camera))
    observation = builder.AddSystem(ImageFeatureSystem(dt=config.dt))
    strategy = builder.AddSystem(
        HeuristicStrategySystem(
            BeihangBaselineStrategy(vehicle=config.vehicle, config=config.strategy)
        )
    )
    scoring = builder.AddSystem(CaptureStatusSystem(config))
    logger = builder.AddSystem(TrialLogger(dt=config.dt))

    builder.Connect(plant.GetOutputPort("vehicle_state"),
                    scene.GetInputPort("vehicle_state"))
    builder.Connect(target.GetOutputPort("target_state"),
                    scene.GetInputPort("target_state"))

    builder.Connect(scene.GetOutputPort("scene"),
                    camera.GetInputPort("scene"))
    builder.Connect(camera.GetOutputPort("image_feature"),
                    observation.GetInputPort("image_feature"))
    builder.Connect(plant.GetOutputPort("vehicle_state"),
                    observation.GetInputPort("vehicle_state"))

    builder.Connect(observation.GetOutputPort("observation"),
                    strategy.GetInputPort("observation"))
    builder.Connect(strategy.GetOutputPort("ctbr_cmd"),
                    plant.GetInputPort("ctbr_cmd"))

    builder.Connect(scene.GetOutputPort("scene"),
                    scoring.GetInputPort("scene"))
    builder.Connect(camera.GetOutputPort("image_feature"),
                    scoring.GetInputPort("image_feature"))
    builder.Connect(strategy.GetOutputPort("ctbr_cmd"),
                    scoring.GetInputPort("ctbr_cmd"))

    builder.Connect(plant.GetOutputPort("vehicle_state"),
                    logger.GetInputPort("vehicle_state"))
    builder.Connect(target.GetOutputPort("target_state"),
                    logger.GetInputPort("target_state"))
    builder.Connect(camera.GetOutputPort("image_feature"),
                    logger.GetInputPort("image_feature"))
    builder.Connect(observation.GetOutputPort("observation"),
                    logger.GetInputPort("observation"))
    builder.Connect(strategy.GetOutputPort("ctbr_cmd"),
                    logger.GetInputPort("ctbr_cmd"))
    builder.Connect(scoring.GetOutputPort("metrics"),
                    logger.GetInputPort("metrics"))

    return builder.Build(), logger

