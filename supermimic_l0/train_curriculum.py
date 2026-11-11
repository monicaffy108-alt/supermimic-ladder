"""L0 training: Pendulum-v1 + PPO + a curriculum-scheduled amplitude.

Usage:
    cd supermimic_l0
    python train_curriculum.py

Takes about 3 minutes on CPU (~2000 fps).
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import tracking_env as te

RAMP_STEPS = 250_000      # amplitude ramps from AMP_START to AMP_FINAL over this many steps
TOTAL_STEPS = 400_000
MODEL_PATH = 'models/l0_ppo_curriculum.zip'


class AmpCurriculum(BaseCallback):
    """Nudges the amplitude toward AMP_FINAL every step, so the policy never
    collapses back to parking after being trained on an easy version."""

    def __init__(self):
        super().__init__()
        self._last_mark = -1

    def _on_step(self):
        p = min(1.0, self.num_timesteps / RAMP_STEPS)
        te.set_amp(te.AMP_START + (te.AMP_FINAL - te.AMP_START) * p)
        mark = self.num_timesteps // 50_000
        if mark != self._last_mark:
            self._last_mark = mark
            print(f'[curriculum] step={self.num_timesteps:>7d}  amplitude={te.get_amp():.3f}')
        return True


def main():
    peak = te.required_peak_torque(te.AMP_FINAL)
    print(f'Final amplitude AMP={te.AMP_FINAL}  frequency={te.FREQ} Hz')
    print(f'Peak torque for exact tracking : {peak:.3f} / limit {te.MAX_TORQUE:.1f} -> '
          + ('feasible' if peak <= te.MAX_TORQUE else 'INFEASIBLE'))
    print()

    te.set_amp(te.AMP_START)
    # DummyVecEnv: all envs live in this process, so the curriculum callback can
    # change the global amplitude and have it take effect immediately.
    env = make_vec_env(te.make_env, n_envs=4, vec_env_cls=DummyVecEnv)
    model = PPO('MlpPolicy', env,
                n_steps=1024, batch_size=64,
                learning_rate=3e-4, gamma=0.99,
                ent_coef=0.0,
                device='cpu', verbose=0)
    model.learn(total_timesteps=TOTAL_STEPS, callback=AmpCurriculum())

    te.set_amp(te.AMP_FINAL)
    model.save(MODEL_PATH)
    print(f'\nModel saved: {MODEL_PATH}')


if __name__ == '__main__':
    main()
