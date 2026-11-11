"""L0 训练：Pendulum-v1 + PPO + 课程式振幅。

用法：
    cd supermimic_l0
    python train_curriculum.py

耗时约 3 分钟（CPU，~2000 fps）。
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import tracking_env as te

RAMP_STEPS = 250_000      # 前 250k 步振幅从 AMP_START 线性涨到 AMP_FINAL
TOTAL_STEPS = 400_000
MODEL_PATH = 'models/l0_ppo_curriculum.zip'


class AmpCurriculum(BaseCallback):
    """每个 step 把振幅往 AMP_FINAL 推一点，避免策略在易任务上过拟合后塌回躺平。"""

    def __init__(self):
        super().__init__()
        self._last_mark = -1

    def _on_step(self):
        p = min(1.0, self.num_timesteps / RAMP_STEPS)
        te.set_amp(te.AMP_START + (te.AMP_FINAL - te.AMP_START) * p)
        mark = self.num_timesteps // 50_000
        if mark != self._last_mark:
            self._last_mark = mark
            print(f'[curriculum] step={self.num_timesteps:>7d}  当前振幅={te.get_amp():.3f}')
        return True


def main():
    peak = te.required_peak_torque(te.AMP_FINAL)
    print(f'最终振幅 AMP={te.AMP_FINAL}  频率={te.FREQ} Hz')
    print(f'精确跟踪所需峰值力矩 : {peak:.3f} / 上限 {te.MAX_TORQUE:.1f} → '
          + ('可行' if peak <= te.MAX_TORQUE else '不可行'))
    print()

    te.set_amp(te.AMP_START)
    # DummyVecEnv：所有 env 在同一进程，课程回调改全局振幅才能立刻生效
    env = make_vec_env(te.make_env, n_envs=4, vec_env_cls=DummyVecEnv)
    model = PPO('MlpPolicy', env,
                n_steps=1024, batch_size=64,
                learning_rate=3e-4, gamma=0.99,
                ent_coef=0.0,
                device='cpu', verbose=0)
    model.learn(total_timesteps=TOTAL_STEPS, callback=AmpCurriculum())

    te.set_amp(te.AMP_FINAL)
    model.save(MODEL_PATH)
    print(f'\n模型已保存: {MODEL_PATH}')


if __name__ == '__main__':
    main()
