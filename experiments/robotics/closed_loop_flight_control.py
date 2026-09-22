"""
COMPUTATIONAL COUPLING — Embodied Closed-Loop Autonomous Flight Control Benchmark
Simulates autonomous micro-UAV quadrotor flight through a turbulent gust corridor:
  - 2D horizontal flight dynamics with severe stochastic wind shear (Ornstein-Uhlenbeck)
  - 8-channel event proximity rangefinders
  - Persistent stateful recurrent execution across time steps (maintaining membrane potentials)
  - Controllers evaluated:
      1. ActiveDendriticCEN (AD-CEN, two-compartment active dendritic control)
      2. CoupledEventNet (CEN, motif-routed sparse core)
      3. Connectome Reservoir (conn2res / C. elegans connectome)
      4. Dense SNN (fully recurrent point LIF)
      5. Classical PID Autopilot
  - Measures: Collision-Free Rate (%), Lateral Tracking RMSE (m), SynOps/Flight, and LiPo Battery Endurance.
"""

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.connectome import load_celegans_connectome
from src.architecture.coupled_event_net import CoupledEventNet, CENConfig
from src.architecture.active_dendritic_cen import ActiveDendriticCEN, ActiveDendriticConfig
from src.architecture.baselines import DenseSNN, ConnectomeReservoirNet


@dataclass
class FlightSimulationConfig:
    corridor_length_m: float = 50.0
    corridor_width_m: float = 8.0
    num_obstacles: int = 12
    obstacle_radius_m: float = 0.35
    dt_s: float = 0.02            # 20 ms control step (50 Hz control loop)
    max_steps: int = 600          # Up to 12 seconds flight
    forward_speed_mps: float = 4.5
    gust_std_dev: float = 1.8     # Wind gust force amplitude (m/s)
    num_sensor_rays: int = 8
    battery_capacity_mah: float = 250.0  # 250 mAh micro-drone battery (3.7V = 3,330 J)
    propulsion_power_w: float = 6.2      # Mechanical motor hover/forward power


class TurbulentFlightEnvironment:
    """Simulates 2D horizontal quadrotor navigation with obstacle avoidance in wind gusts."""

    def __init__(self, cfg: FlightSimulationConfig, seed: int = 42):
        self.cfg = cfg
        np.random.seed(seed)
        # Position obstacles along the corridor with clear corridors
        self.obstacle_x = np.linspace(6.0, cfg.corridor_length_m - 6.0, cfg.num_obstacles)
        # Offset obstacles alternating left and right to require slalom navigation
        offsets = np.array([-1.5, 1.8, -1.2, 2.0, -1.8, 1.4, -2.0, 1.2, -1.5, 1.7, -1.3, 1.5])
        self.obstacle_y = offsets[:cfg.num_obstacles]
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.vy = 0.0
        self.step_idx = 0
        self.gust_state = 0.0
        self.collided = False
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        angles = np.linspace(-np.pi / 3.0, np.pi / 3.0, self.cfg.num_sensor_rays)
        obs = np.zeros(self.cfg.num_sensor_rays, dtype=np.float32)

        for idx, ang in enumerate(angles):
            ray_dx = np.cos(ang)
            ray_dy = np.sin(ang)

            min_dist = 8.0
            for ox, oy in zip(self.obstacle_x, self.obstacle_y):
                d_ox = ox - self.x
                d_oy = oy - self.y
                proj = d_ox * ray_dx + d_oy * ray_dy
                if proj > 0:
                    perp_dist = np.abs(d_ox * (-ray_dy) + d_oy * ray_dx)
                    if perp_dist < self.cfg.obstacle_radius_m + 0.10:
                        dist = max(0.05, proj - self.cfg.obstacle_radius_m)
                        if dist < min_dist:
                            min_dist = dist

            # Wall boundaries
            wall_dist_pos = (self.cfg.corridor_width_m / 2.0 - self.y) / max(1e-3, ray_dy) if ray_dy > 0 else 8.0
            wall_dist_neg = (-self.cfg.corridor_width_m / 2.0 - self.y) / min(-1e-3, ray_dy) if ray_dy < 0 else 8.0
            min_dist = min(min_dist, wall_dist_pos if wall_dist_pos > 0 else 8.0, wall_dist_neg if wall_dist_neg > 0 else 8.0)

            # Inverted proximity: 1.0 at contact, 0.0 at >= 5m
            obs[idx] = float(np.clip(1.0 - (min_dist / 5.0), 0.0, 1.0))

        return obs

    def step(self, lateral_cmd: float) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        self.step_idx += 1

        # Ornstein-Uhlenbeck wind gust process
        self.gust_state = 0.88 * self.gust_state + np.random.normal(0.0, self.cfg.gust_std_dev * 0.35)

        thrust_force = np.clip(lateral_cmd, -5.0, 5.0)
        ay = thrust_force + self.gust_state
        self.vy = 0.90 * self.vy + ay * self.cfg.dt_s
        self.y += self.vy * self.cfg.dt_s
        self.x += self.cfg.forward_speed_mps * self.cfg.dt_s

        # Collision check
        for ox, oy in zip(self.obstacle_x, self.obstacle_y):
            dist = np.sqrt((self.x - ox) ** 2 + (self.y - oy) ** 2)
            if dist < (self.cfg.obstacle_radius_m + 0.12):
                self.collided = True

        if abs(self.y) >= self.cfg.corridor_width_m / 2.0:
            self.collided = True

        done = (self.x >= self.cfg.corridor_length_m) or (self.step_idx >= self.cfg.max_steps) or self.collided
        info = {
            "x": self.x,
            "y": self.y,
            "collided": self.collided,
            "success": (self.x >= self.cfg.corridor_length_m) and not self.collided,
        }
        return self._get_observation(), done, info


class StatefulCENController:
    """Stateful CoupledEventNet controller maintaining membrane potentials across time."""

    def __init__(self, num_neurons: int = 64):
        self.cfg = CENConfig(input_dim=8, num_neurons=num_neurons, output_dim=1, cffl_fraction=0.35, iffl_fraction=0.25, rip_fraction=0.20)
        self.model = CoupledEventNet(self.cfg)
        self.N = num_neurons
        self.reset()

        # Initialize bio-grounded sensory weights (left obstacle triggers positive thrust to steer away)
        with torch.no_grad():
            self.model.w_in[:num_neurons//2, :4] = 1.4   # Left sensors excite left population
            self.model.w_in[num_neurons//2:, 4:] = 1.4   # Right sensors excite right population
            self.model.w_out.weight[0, :num_neurons//2] = 0.60   # Left obstacle triggers positive rightward steer
            self.model.w_out.weight[0, num_neurons//2:] = -0.60  # Right obstacle triggers negative leftward steer

    def reset(self):
        self.v = torch.zeros(self.N)
        self.i_syn = torch.zeros(self.N)
        self.s_prev = torch.zeros(self.N)

    def step(self, obs: np.ndarray) -> Tuple[float, float, float]:
        x = torch.tensor(obs, dtype=torch.float32)
        W_rec = self.model.get_effective_recurrent_weights()

        with torch.no_grad():
            i_in = self.model.w_in @ x
            i_rec = W_rec @ self.s_prev
            self.i_syn = self.i_syn * self.model.alpha_s + (i_in + i_rec)
            self.v = self.v * self.model.alpha_m + self.i_syn - (self.cfg.v_thresh - self.cfg.v_reset) * self.s_prev

            s = (self.v >= self.cfg.v_thresh).float()
            self.s_prev = s

            out = self.model.w_out(self.v)
            cmd = float(out.item()) * 4.0

            n_spikes = float(s.sum().item())
            synops = float((s @ self.model.out_degs).item())
            energy_nj = synops * 0.020 + n_spikes * 0.005

        return cmd, synops, energy_nj


class StatefulADCENController:
    """Stateful Active Dendritic CEN controller with NMDA plateaus for gust holding."""

    def __init__(self, num_neurons: int = 64):
        self.cfg = ActiveDendriticConfig(input_dim=8, num_neurons=num_neurons, output_dim=1, tau_plateau_ms=180.0, g_nmda=0.40, theta_dend=0.35)
        self.model = ActiveDendriticCEN(self.cfg)
        self.N = num_neurons
        self.reset()

        with torch.no_grad():
            self.model.w_in[:num_neurons//2, :4] = 1.4
            self.model.w_in[num_neurons//2:, 4:] = 1.4
            self.model.w_out.weight[0, :num_neurons//2] = 0.60
            self.model.w_out.weight[0, num_neurons//2:] = -0.60

    def reset(self):
        self.v_dend = torch.zeros(self.N)
        self.v_soma = torch.zeros(self.N)
        self.i_syn_dend = torch.zeros(self.N)
        self.i_syn_soma_inh = torch.zeros(self.N)
        self.s_soma_prev = torch.zeros(self.N)

    def step(self, obs: np.ndarray) -> Tuple[float, float, float]:
        x = torch.tensor(obs, dtype=torch.float32)
        W_rec = self.model.get_effective_recurrent_weights()
        W_rec_exc = W_rec * self.model.is_exc_pre.unsqueeze(0)
        W_rec_inh = -W_rec * self.model.is_inh_pre.unsqueeze(0)

        with torch.no_grad():
            i_in = self.model.w_in @ x
            i_rec_exc = W_rec_exc @ self.s_soma_prev
            i_rec_inh = W_rec_inh @ self.s_soma_prev

            self.i_syn_dend = self.i_syn_dend * self.model.alpha_syn + (i_in + i_rec_exc)
            self.i_syn_soma_inh = self.i_syn_soma_inh * self.model.alpha_syn + i_rec_inh

            nmda_gate = torch.sigmoid((self.v_dend - self.cfg.theta_dend) * 12.0)
            i_nmda = self.cfg.g_nmda * nmda_gate * (self.cfg.v_plateau - self.v_dend)
            alpha_d_eff = self.model.alpha_dend * (1.0 - nmda_gate) + self.model.alpha_plateau * nmda_gate

            self.v_dend = alpha_d_eff * self.v_dend + self.i_syn_dend + i_nmda
            i_axial = self.cfg.g_axial * (self.v_dend - self.v_soma)

            shunt = torch.clamp(self.cfg.shunting_efficiency * torch.tanh(self.i_syn_soma_inh * 2.0), 0.0, 0.99)
            v_soma_decayed = self.model.alpha_soma * self.v_soma * (1.0 - shunt) + i_axial * (1.0 - shunt)
            self.v_soma = v_soma_decayed - (self.cfg.v_thresh_soma - self.cfg.v_reset_soma) * self.s_soma_prev

            s_soma = (self.v_soma >= self.cfg.v_thresh_soma).float()
            self.s_soma_prev = s_soma

            out = self.model.w_out(self.v_soma)
            cmd = float(out.item()) * 4.0

            n_spikes = float(s_soma.sum().item())
            synops = float((s_soma @ self.model.out_degs).item())
            energy_nj = synops * 0.020 + n_spikes * 0.005

        return cmd, synops, energy_nj


class StatefulDenseController:
    """Stateful Dense SNN controller (standard all-to-all connectivity)."""

    def __init__(self, num_neurons: int = 64):
        self.N = num_neurons
        self.model = DenseSNN(input_dim=8, num_neurons=num_neurons, output_dim=1)
        self.alpha_m = float(np.exp(-20.0 / 20.0)) # dt=20ms, tau=20ms
        self.alpha_s = float(np.exp(-20.0 / 5.0))
        self.reset()

        with torch.no_grad():
            self.model.w_in.weight[:num_neurons//2, :4] = 1.4
            self.model.w_in.weight[num_neurons//2:, 4:] = 1.4
            self.model.w_out.weight[0, :num_neurons//2] = 0.60
            self.model.w_out.weight[0, num_neurons//2:] = -0.60

    def reset(self):
        self.v = torch.zeros(self.N)
        self.i_syn = torch.zeros(self.N)
        self.s_prev = torch.zeros(self.N)

    def step(self, obs: np.ndarray) -> Tuple[float, float, float]:
        x = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            i_in = self.model.w_in(x)
            i_rec = self.model.w_rec(self.s_prev)
            self.i_syn = self.i_syn * self.alpha_s + (i_in + i_rec)
            self.v = self.v * self.alpha_m + self.i_syn - 1.0 * self.s_prev

            s = (self.v >= 1.0).float()
            self.s_prev = s

            out = self.model.w_out(self.v)
            cmd = float(out.item()) * 4.0

            n_spikes = float(s.sum().item())
            synops = float((s @ self.model.out_degs).item())
            energy_nj = synops * 0.020 + n_spikes * 0.005

        return cmd, synops, energy_nj


class StatefulReservoirController:
    """Stateful Connectome Reservoir Controller (conn2res / C. elegans connectome)."""

    def __init__(self):
        g_ce = load_celegans_connectome()
        adj_ce, _ = g_ce.get_adjacency_matrix(weighted=True, signed=True)
        self.model = ConnectomeReservoirNet(adjacency_matrix=adj_ce, input_dim=8, output_dim=1, spectral_radius=0.90)
        self.N = self.model.num_neurons
        self.reset()

        with torch.no_grad():
            self.model.W_in[:self.N//2, :4] = 1.4
            self.model.W_in[self.N//2:, 4:] = 1.4
            self.model.w_out.weight[0, :self.N//2] = 0.60
            self.model.w_out.weight[0, self.N//2:] = -0.60

    def reset(self):
        self.v = torch.zeros(self.N)
        self.i_syn = torch.zeros(self.N)
        self.s_prev = torch.zeros(self.N)

    def step(self, obs: np.ndarray) -> Tuple[float, float, float]:
        x = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            i_in = F.linear(x, self.model.W_in)
            i_rec = F.linear(self.s_prev, self.model.W_rec)
            self.i_syn = self.i_syn * self.model.alpha_s + (i_in + i_rec)
            v_new = self.v * self.model.alpha_m + self.i_syn * (1.0 - self.model.alpha_m)

            s = (v_new >= self.model.v_thresh).float()
            self.v = v_new * (1.0 - s)
            self.s_prev = s

            out = self.model.w_out(self.v)
            cmd = float(out.item()) * 4.0

            n_spikes = float(s.sum().item())
            synops = float((s @ self.model.out_degs).item())
            energy_nj = synops * 0.020 + n_spikes * 0.005

        return cmd, synops, energy_nj


def evaluate_flight_controller(
    controller_name: str,
    controller_cls: Any,
    num_trials: int = 15,
) -> Dict[str, Any]:
    cfg = FlightSimulationConfig()
    successes = []
    tracking_errors = []
    synops_per_trial = []
    energies_nj_per_trial = []

    for trial in range(num_trials):
        env = TurbulentFlightEnvironment(cfg, seed=2000 + trial * 11)
        obs = env.reset()
        controller = controller_cls() if callable(controller_cls) else None
        if hasattr(controller, "reset"):
            controller.reset()

        done = False
        trial_y = []
        trial_synops = 0.0
        trial_energy = 0.0

        while not done:
            trial_y.append(env.y)
            if controller_name == "Classical PID Autopilot":
                # Reactive proportional-derivative wall/obstacle repulsor
                left_rep = float(np.sum(obs[:4]))
                right_rep = float(np.sum(obs[4:]))
                cmd = (left_rep - right_rep) * 3.8 - 0.4 * env.vy
                synops, energy = 0.0, 0.0
            else:
                cmd, synops, energy = controller.step(obs)
                # Dampen lateral velocity for flight stability
                cmd = cmd - 0.25 * env.vy

            trial_synops += synops
            trial_energy += energy
            obs, done, info = env.step(cmd)

        successes.append(1.0 if info["success"] else 0.0)
        tracking_errors.append(float(np.sqrt(np.mean(np.array(trial_y) ** 2))))
        synops_per_trial.append(trial_synops)
        energies_nj_per_trial.append(trial_energy)

    mean_success = float(np.mean(successes)) * 100.0
    mean_rmse = float(np.mean(tracking_errors))
    mean_synops = float(np.mean(synops_per_trial))
    mean_energy_nj = float(np.mean(energies_nj_per_trial))

    flight_time_s = cfg.corridor_length_m / cfg.forward_speed_mps # ~11s
    compute_power_w = (mean_energy_nj * 1e-9) / max(1e-4, flight_time_s)
    total_power_w = cfg.propulsion_power_w + compute_power_w
    battery_joules = (cfg.battery_capacity_mah / 1000.0) * 3.7 * 3600.0
    endurance_min = (battery_joules / total_power_w) / 60.0

    return {
        "controller": controller_name,
        "success_rate_pct": mean_success,
        "rmse_m": mean_rmse,
        "mean_synops": mean_synops,
        "energy_proxy_nj": mean_energy_nj,
        "compute_power_mw": compute_power_w * 1000.0,
        "flight_endurance_min": endurance_min,
    }


def run_robotics_benchmark():
    print("=" * 85)
    print("COMPUTATIONAL COUPLING — Closed-Loop Neuro-Robotic Autonomous Flight Control")
    print("=" * 85)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    controllers = [
        ("ActiveDendriticCEN (AD-CEN)", StatefulADCENController),
        ("CoupledEventNet (CEN)", StatefulCENController),
        ("Classical PID Autopilot", None),
        ("Connectome Reservoir (conn2res)", StatefulReservoirController),
        ("Dense SNN", StatefulDenseController),
    ]

    benchmark_data = []
    print("\n[*] Evaluating Controllers Across 15 Turbulent Monte Carlo Flights...")

    for name, cls in controllers:
        res = evaluate_flight_controller(name, cls, num_trials=15)
        benchmark_data.append(res)
        print(f"  {name:<30}: Success = {res['success_rate_pct']:5.1f}% | RMSE = {res['rmse_m']:4.2f}m | SynOps = {res['mean_synops']:7.1f} | Endurance = {res['flight_endurance_min']:4.2f} min")

    df = pd.DataFrame(benchmark_data)
    csv_path = os.path.join(results_dir, "closed_loop_robotics_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved Closed-Loop Robotics CSV to {csv_path}")

    json_path = os.path.join(results_dir, "closed_loop_robotics_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"[+] Saved Closed-Loop Robotics JSON to {json_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_robotics_benchmark()
