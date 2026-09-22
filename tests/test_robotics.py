"""
Unit tests for closed-loop neuro-robotic flight control environment and stateful controllers.
"""

import pytest
import numpy as np
from experiments.robotics.closed_loop_flight_control import (
    FlightSimulationConfig,
    TurbulentFlightEnvironment,
    StatefulCENController,
    StatefulADCENController,
)


def test_flight_environment_step_and_observation():
    """Verify flight environment reset, step, and ray rangefinder physics."""
    cfg = FlightSimulationConfig(corridor_length_m=20.0, num_obstacles=5, max_steps=50)
    env = TurbulentFlightEnvironment(cfg, seed=42)

    obs = env.reset()
    assert obs.shape == (cfg.num_sensor_rays,)
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)

    # Step forward with 0 lateral thrust
    next_obs, done, info = env.step(0.0)
    assert next_obs.shape == (cfg.num_sensor_rays,)
    assert info["x"] > 0.0
    assert "collided" in info
    assert "success" in info


def test_stateful_cen_controller_step():
    """Verify stateful CEN controller step returns control action and SynOps."""
    ctrl = StatefulCENController(num_neurons=32)
    obs = np.array([0.2, 0.5, 0.1, 0.0, 0.0, 0.1, 0.4, 0.8], dtype=np.float32)

    cmd, synops, energy = ctrl.step(obs)
    assert isinstance(cmd, float)
    assert synops >= 0.0
    assert energy >= 0.0


def test_stateful_ad_cen_controller_step():
    """Verify stateful Active Dendritic CEN controller step."""
    ctrl = StatefulADCENController(num_neurons=32)
    obs = np.array([0.9, 0.8, 0.4, 0.0, 0.0, 0.0, 0.1, 0.2], dtype=np.float32)

    cmd, synops, energy = ctrl.step(obs)
    assert isinstance(cmd, float)
    assert synops >= 0.0
    assert energy >= 0.0
