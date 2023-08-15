import attr
import numpy as np

import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction

@habitat.registry.register_task_action
class PAUSE(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "pause"

    def step(self, *args, **kwargs):
        return self._sim.get_observations_at()  # type: ignore
    

@habitat.registry.register_action_space_configuration
class TURN90(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TurnLeft90] = habitat_sim.ActionSpec(
                "turn_left_90",
                habitat_sim.ActuationSpec(amount=90),
            ),
        config[HabitatSimActions.TurnRight90] = habitat_sim.ActionSpec(
                "turn_right_90",
                habitat_sim.ActuationSpec(amount=90),
            )

        return config
    
@habitat.registry.register_task_action
class TurnRight90(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "turn_right_90"

    def step(self, *args, **kwargs):
        # return self._sim.step(HabitatSimActions.TurnRight90)
        for _ in range(2):
            self._sim.step(HabitatSimActions.TURN_RIGHT)
        return self._sim.step(HabitatSimActions.TURN_RIGHT)
    
    
@habitat.registry.register_task_action
class TurnLeft90(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "turn_left_90"

    def step(self, *args, **kwargs):
        # return self._sim.step(HabitatSimActions.TurnLeft90)
        for _ in range(2):
            self._sim.step(HabitatSimActions.TURN_LEFT)
        return self._sim.step(HabitatSimActions.TURN_LEFT)