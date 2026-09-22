"""
COMPUTATIONAL COUPLING — Neuromorphic Architecture & Baselines Package
"""

from .coupled_event_net import (
    CENConfig,
    CoupledEventNet,
    SurrogateHeaviside,
)
from .baselines import (
    DenseSNN,
    ConnectomeReservoirNet,
)

__all__ = [
    "CENConfig",
    "CoupledEventNet",
    "SurrogateHeaviside",
    "DenseSNN",
    "ConnectomeReservoirNet",
]
