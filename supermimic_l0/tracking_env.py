"""L0 跟踪环境：把 Pendulum-v1 改造成「跟着参考轨迹摆动」的任务。

三个关键设计（都是踩过坑才定下来的）：
1. 参考相位必须进观测。只给 [cosθ, sinθ, θ̇] 时，策略无法区分
   「t=1.25s 目标是 +0.6」和「t=3.75s 目标是 −0.6」，只能收敛到转圈。
2. 参考振幅必须在力矩可达范围内。Pendulum 动力学 θ̈ = 15·sinθ + 3u 无阻尼，
   静止撑住角度需要 u = 5·sinθ，而 |u| ≤ 2 → 最大可静止角约 0.41 rad。
3. 振幅要课程式增长。直接从 0.3 开始，策略会收敛到「躺在 0 不动」的局部最优，
   因为躺平的每步代价只有 A²/2。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

DT = 0.05
FREQ = 0.2
AMP_START = 0.05
AMP_FINAL = 0.3
MAX_TORQUE = 2.0

# 当前生效振幅。DummyVecEnv 下同进程共享，供课程回调实时修改。
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
    """把角度差折叠到 [-π, π]，避免跨越 ±π 时出现 2π 的虚假误差。"""
    return np.arctan2(np.sin(a), np.cos(a))


def required_peak_torque(amp, freq=FREQ, dt=DT):
    """精确跟踪该参考轨迹所需的峰值力矩，用于检查任务是否物理可行。"""
    w = 2 * np.pi * freq
    ts = np.arange(0, 2 * np.pi / w, dt)
    u = (-(amp * w ** 2) * np.sin(w * ts)
         - 15 * np.sin(amp * np.sin(w * ts))) / 3
    return float(np.abs(u).max())


class TrackingPendulum(gym.Wrapper):
    """观测 = [cosθ, sinθ, θ̇, θ_ref, θ̇_ref]，共 5 维。"""

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
        # 完全覆盖原生 reward：原生奖励鼓励停在顶端静止，与跟踪目标冲突
        r = -(err ** 2) - 0.001 * np.sum(np.square(action))
        info.update(theta=theta, ref=theta_ref(self.t), err=err, t=self.t)
        return self._aug(obs), r, term, trunc, info


def make_env():
    return TrackingPendulum(gym.make('Pendulum-v1'))
