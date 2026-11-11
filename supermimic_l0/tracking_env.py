"""L0 tracking environment: turns Pendulum-v1 into "swing along a reference".

Three design choices, each of which came from a failure mode we hit:
1. The reference phase must be part of the observation. With only
   [cos, sin, theta_dot] the policy cannot tell "at t=1.25s the target is +0.6"
   apart from "at t=3.75s the target is -0.6", so it collapses to spinning.
2. The reference amplitude must stay within the torque budget. Pendulum dynamics
   are theta_ddot = 15*sin(theta) + 3u with no damping, so holding an angle
   requires u = 5*sin(theta); with |u| <= 2 the largest holdable angle is
   about 0.41 rad.
3. The amplitude must ramp up as a curriculum. Starting at 0.3 directly lets the
   policy settle into the "park at zero" local optimum, because parking only
   costs A^2/2 per step.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

DT = 0.05
FREQ = 0.2
AMP_START = 0.05
AMP_FINAL = 0.3
MAX_TORQUE = 2.0

# Amplitude currently in effect. Under DummyVecEnv every env shares this process,
# so the curriculum callback can update it live.
_CUR_AMP = AMP_START


def set_amp(v):
    global _CUR_AMP
    _CUR_AMP = float(v)


def get_amp():
    return _CUR_AMP


def theta_ref(t):
    return _CUR_AMP * np.sin(2 * np.pi * FREQ * t)


def dtheta_ref(t):
    return _CUR_AMP * 2 * np.pi * FREQ * np.cos(2 * np.pi * FREQ * t)


def wrap(a):
    """Wrap an angular difference into [-pi, pi] so crossing the boundary
    does not produce a spurious 2*pi error."""
    return np.arctan2(np.sin(a), np.cos(a))


def required_peak_torque(amp, freq=FREQ, dt=DT):
    """Peak torque needed to track this reference exactly; used to check that
    the task is physically feasible before training starts."""
    w = 2 * np.pi * freq
    ts = np.arange(0, 2 * np.pi / w, dt)
    u = (-(amp * w ** 2) * np.sin(w * ts)
         - 15 * np.sin(amp * np.sin(w * ts))) / 3
    return float(np.abs(u).max())


class TrackingPendulum(gym.Wrapper):
    """Observation = [cos, sin, theta_dot, ref, ref_dot], 5 dimensions."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.t = 0.0

    def _aug(self, obs):
        return np.array([obs[0], obs[1], obs[2],
                         theta_ref(self.t), dtheta_ref(self.t)], dtype=np.float32)

    def reset(self, **kwargs):
        self.t = 0.0
        obs, info = self.env.reset(**kwargs)
        return self._aug(obs), info

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        self.t += DT
        theta = np.arctan2(obs[1], obs[0])
        err = wrap(theta - theta_ref(self.t))
        # Fully overrides the native reward: the native one rewards parking at
        # the top, which directly conflicts with tracking a moving reference.
        r = -(err ** 2) - 0.001 * np.sum(np.square(action))
        info.update(theta=theta, ref=theta_ref(self.t), err=err, t=self.t)
        return self._aug(obs), r, term, trunc, info


def make_env():
    return TrackingPendulum(gym.make('Pendulum-v1'))
