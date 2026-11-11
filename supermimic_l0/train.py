"""
SuperMimic L0 —— Pendulum-v1 motion imitation sanity check
==========================================================

任务：让 PPO agent 在 Pendulum-v1 上跟踪一条手工生成的参考轨迹。

核心思路：
  1. 手工生成一条周期性参考轨迹（theta, theta_dot, 隐含 tau 由 env 动力学决定）
  2. 把 "跟踪参考轨迹" 编码到 reward 里，而非改 env
  3. 用 stable-baselines3 PPO 训练
  4. 训练完画曲线 + 生成跟踪动画

运行：
  python train.py           # 训练并保存
  python train.py --eval    # 加载已保存模型做评估
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")  # headless 友好
import matplotlib.pyplot as plt


def _lazy_import_sb3():
    """sb3 + torch 延迟导入，让 wrapper/参考轨迹可以独立测试。"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    return PPO, DummyVecEnv, CallbackList, CheckpointCallback, Monitor


# ---------------------------------------------------------------------------
# 1. 手工生成参考轨迹
# ---------------------------------------------------------------------------

def make_reference_trajectory(
    duration: float = 10.0,   # 秒
    dt: float = 0.05,         # Pendulum-v1 默认 dt
    pattern: str = "sine",
    seed: int = 0,
) -> dict:
    """生成 Pendulum 的参考轨迹。

    Pendulum 状态是 [cos(theta), sin(theta), theta_dot]，
    我们生成一条平滑的周期性轨迹，让 agent 去跟踪。
    """
    t = np.arange(0.0, duration, dt)
    np.random.seed(seed)

    if pattern == "sine":
        # 正弦摆动：从下垂开始，摆到上方再回来
        theta_ref = np.pi * 0.4 * np.sin(2 * np.pi * t * 0.5)  # ±72°
        theta_dot_ref = np.pi * 0.4 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * t * 0.5)
    elif pattern == "swingup":
        # 摆起+平衡：从下垂(pi)摆到上方(0)
        theta_ref = np.where(
            t < 2.0,
            np.pi - 0.5 * np.sin(np.pi * t / 2.0),   # 前 2s 摆起
            0.1 * np.sin(2 * np.pi * t * 1.0),         # 后段保持在上方微调
        )
        theta_dot_ref = np.gradient(theta_ref, dt)
    elif pattern == "random_walk":
        # 随机游走式平滑轨迹
        steps = len(t)
        noise = np.cumsum(np.random.randn(steps) * 0.1)
        theta_ref = np.clip(noise, -np.pi * 0.8, np.pi * 0.8)
        theta_dot_ref = np.gradient(theta_ref, dt)
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    # 打包成标准格式
    cos_ref = np.cos(theta_ref)
    sin_ref = np.sin(theta_ref)

    return {
        "t": t,
        "theta": theta_ref,
        "cos": cos_ref,
        "sin": sin_ref,
        "theta_dot": theta_dot_ref,
        "dt": dt,
        "pattern": pattern,
    }


# ---------------------------------------------------------------------------
# 2. 跟踪奖励 Wrapper
# ---------------------------------------------------------------------------

class MotionImitationWrapper(gym.Wrapper):
    """在 Pendulum-v1 原奖励基础上，叠加跟踪参考轨迹的误差惩罚。

    Pendulum-v1 原 reward 是 (theta² + 0.1*theta_dot² + 0.001*action²) 的负值，
    本身鼓励摆杆停在上方静止，但没有显式跟踪参考轨迹的机制。

    这里在每个 step 根据当前 env.sim.state (theta, theta_dot) 和 ref 的对应时刻
    计算跟踪误差，按比例加到 reward 里。
    """

    def __init__(self, env, ref: dict, pos_weight: float = 1.0, vel_weight: float = 0.1):
        super().__init__(env)
        self.ref = ref
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self._step_idx = 0

    def reset(self, **kwargs):
        self._step_idx = 0
        # 初始状态对齐到参考轨迹的起点
        obs, info = self.env.reset(**kwargs)
        theta0 = self.ref["theta"][0]
        theta_dot0 = self.ref["theta_dot"][0]
        # gymnasium PendulumEnv 暴露 .state 属性，直接赋值
        try:
            self.env.unwrapped.state = np.array([theta0, theta_dot0], dtype=np.float32)
        except Exception:
            pass
        return obs, info

    def _get_pendulum_state(self):
        """从 PendulumEnv 拿到 [theta, theta_dot]。

        gymnasium 1.4+ 直接暴露 .state；旧版是 .sim.state。
        """
        u = self.env.unwrapped
        try:
            s = u.state  # gymnasium 1.4+
            return float(s[0]), float(s[1])
        except Exception:
            pass
        try:
            s = u.sim.state  # 旧版 Box2D v3
            return float(s[0]), float(s[1])
        except Exception:
            pass
        # 退化：从 obs 反推
        cos_t, sin_t, theta_dot = self.env.unwrapped._get_obs()
        return float(np.arctan2(sin_t, cos_t)), float(theta_dot)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """计算带 wrap-around 的角度差，范围 [-pi, pi]。"""
        return float(np.arctan2(np.sin(a - b), np.cos(a - b)))

    def step(self, action):
        obs, _raw_reward, terminated, truncated, info = self.env.step(action)

        # 跟踪误差 —— 完全覆盖原始 Pendulum reward（目标是 imitation，不是摆杆控制）
        idx = min(self._step_idx, len(self.ref["theta"]) - 1)
        theta_cur, theta_dot_cur = self._get_pendulum_state()

        dtheta = self._angle_diff(theta_cur, self.ref["theta"][idx])
        dtheta_dot = theta_dot_cur - self.ref["theta_dot"][idx]
        pos_err = dtheta ** 2
        vel_err = dtheta_dot ** 2
        # tracking reward = -(位置误差 + 速度误差 + 小的 action 正则)
        reward = -(self.pos_weight * pos_err + self.vel_weight * vel_err + 0.001 * float(action[0]) ** 2)

        info["pos_err"] = float(pos_err)
        info["vel_err"] = float(vel_err)
        info["step"] = self._step_idx

        self._step_idx += 1

        # 忽略原始终止条件，只在参考轨迹结束时 truncated
        terminated = False
        if self._step_idx >= len(self.ref["theta"]):
            truncated = True

        return obs, float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# 3. 训练
# ---------------------------------------------------------------------------

def train(args):
    PPO, DummyVecEnv, CallbackList, CheckpointCallback, Monitor = _lazy_import_sb3()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成参考轨迹
    ref = make_reference_trajectory(
        duration=args.duration, dt=0.05, pattern=args.pattern, seed=args.seed
    )
    # 保存参考轨迹
    with open(output_dir / "reference.pkl", "wb") as f:
        pickle.dump(ref, f)

    # 创建 env
    def make_env():
        env = gym.make("Pendulum-v1")
        env = MotionImitationWrapper(env, ref, args.pos_weight, args.vel_weight)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])

    # PPO 配置 —— Pendulum 是低维连续控制，网络不用太大
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=args.seed,
    )

    # 回调：定期保存
    checkpoint_cb = CheckpointCallback(
        save_freq=5000, save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_pendulum",
    )

    print(f"\n=== 开始训练：Pendulum-v1 + PPO + {args.pattern} 跟踪 ===")
    print(f"  总步数: {args.total_timesteps}")
    print(f"  输出目录: {output_dir}")
    print()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([checkpoint_cb]),
        progress_bar=True,
    )

    model.save(str(output_dir / "ppo_pendulum_final"))
    print(f"\n✅ 训练完成，模型已保存到 {output_dir}")

    # 画训练曲线
    plot_training_curves(output_dir)


# ---------------------------------------------------------------------------
# 4. 评估 & 可视化
# ---------------------------------------------------------------------------

def evaluate(args):
    PPO, *_ = _lazy_import_sb3()

    output_dir = Path(args.output_dir)
    ref_path = output_dir / "reference.pkl"
    model_path = output_dir / "ppo_pendulum_final.zip"

    with open(ref_path, "rb") as f:
        ref = pickle.load(f)

    env = gym.make("Pendulum-v1")
    env = MotionImitationWrapper(env, ref, pos_weight=1.0, vel_weight=0.1)

    model = PPO.load(str(model_path))

    obs, _ = env.reset()
    done = False
    traj = {"theta": [], "theta_dot": [], "action": [], "reward": [], "pos_err": [], "vel_err": []}

    print("=== 评估 rollout ===")
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        theta_cur, theta_dot_cur = env._get_pendulum_state()
        traj["theta"].append(theta_cur)
        traj["theta_dot"].append(theta_dot_cur)

        traj["action"].append(float(action[0]))
        traj["reward"].append(float(reward))
        traj["pos_err"].append(info.get("pos_err", 0.0))
        traj["vel_err"].append(info.get("vel_err", 0.0))

        step += 1
        if step % 50 == 0:
            print(f"  step {step}: reward={reward:.2f}  pos_err={info.get('pos_err', 0.0):.4f}")

    env.close()
    plot_tracking_comparison(ref, traj, output_dir)
    print(f"\n✅ 评估完成，图表已保存到 {output_dir}")


# ---------------------------------------------------------------------------
# 5. 画图
# ---------------------------------------------------------------------------

def plot_training_curves(output_dir: Path):
    """从 Monitor 日志画 reward 曲线。"""
    monitor_dirs = sorted(output_dir.glob("*monitor*"))
    log_files = sorted(output_dir.rglob("*.monitor.csv"))
    if not log_files:
        print("⚠️ 没找到 Monitor 日志，跳过训练曲线绘制")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for lf in log_files:
        data = np.genfromtxt(lf, delimiter=",", names=True, skip_header=1)
        returns = data["r"]
        ax.plot(returns, alpha=0.5, label=str(lf.parent.name))
        # 滑动平均
        if len(returns) > 10:
            window = min(50, len(returns) // 10)
            smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
            ax.plot(range(window // 2, window // 2 + len(smoothed)), smoothed,
                    linewidth=2, label=f"smoothed ({lf.parent.name})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("PPO Training Reward — Pendulum-v1 Motion Imitation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_tracking_comparison(ref: dict, traj: dict, output_dir: Path):
    """对比参考轨迹和实际跟踪轨迹。"""
    n = min(len(traj["theta"]), len(ref["theta"]))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t_plot = np.arange(n) * ref["dt"]

    # 角度跟踪
    axes[0].plot(t_plot, ref["theta"][:n], "r--", label="reference", linewidth=2)
    axes[0].plot(t_plot, traj["theta"][:n], "b-", label="tracked", alpha=0.8)
    axes[0].set_ylabel("theta [rad]")
    axes[0].set_title("Pendulum Motion Imitation — Tracking Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 角速度跟踪
    axes[1].plot(t_plot, ref["theta_dot"][:n], "r--", label="ref theta_dot", linewidth=2)
    axes[1].plot(t_plot, traj["theta_dot"][:n], "b-", label="tracked", alpha=0.8)
    axes[1].set_ylabel("theta_dot [rad/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 跟踪误差
    axes[2].plot(t_plot, traj["pos_err"][:n], "g-", label="pos_err")
    axes[2].plot(t_plot, traj["vel_err"][:n], "m-", label="vel_err")
    axes[2].set_ylabel("error")
    axes[2].set_xlabel("time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "tracking_comparison.png", dpi=150)
    plt.close(fig)

    # 打印定量指标 —— 角度必须 wrap-around
    theta_arr = np.array(traj["theta"][:n])
    dtheta = np.arctan2(np.sin(theta_arr - ref["theta"][:n]), np.cos(theta_arr - ref["theta"][:n]))
    pos_rmse = float(np.sqrt(np.mean(dtheta ** 2)))
    vel_rmse = float(np.sqrt(np.mean((np.array(traj["theta_dot"][:n]) - ref["theta_dot"][:n]) ** 2)))
    mean_reward = float(np.mean(traj["reward"]))
    print(f"\n📊 跟踪指标:")
    print(f"   position RMSE : {pos_rmse:.4f} rad")
    print(f"   velocity RMSE : {vel_rmse:.4f} rad/s")
    print(f"   mean step reward : {mean_reward:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SuperMimic L0 — Pendulum motion imitation")
    parser.add_argument("--eval", action="store_true", help="评估模式（加载已有模型）")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--pattern", type=str, default="sine",
                        choices=["sine", "swingup", "random_walk"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--vel-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
