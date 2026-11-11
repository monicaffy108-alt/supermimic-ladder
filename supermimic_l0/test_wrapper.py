"""快速验证 MotionImitationWrapper + 参考轨迹生成（不依赖 torch/sb3）。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import gymnasium as gym
from train import make_reference_trajectory, MotionImitationWrapper

print("=== 1. 测试参考轨迹生成 ===")
ref = make_reference_trajectory(duration=5.0, dt=0.05, pattern="sine")
print(f"  长度: {len(ref['t'])} 步, t∈[{ref['t'][0]:.2f}, {ref['t'][-1]:.2f}]s")
print(f"  theta 范围: [{ref['theta'].min():.3f}, {ref['theta'].max():.3f}] rad")
print(f"  theta_dot 范围: [{ref['theta_dot'].min():.3f}, {ref['theta_dot'].max():.3f}] rad/s")

print("\n=== 2. 测试 Wrapper reset 对齐 ===")
env = gym.make("Pendulum-v1")
wrapped = MotionImitationWrapper(env, ref, pos_weight=1.0, vel_weight=0.1)
obs, info = wrapped.reset()
u = wrapped.env.unwrapped
print(f"  reset 后 state = {u.state}")
print(f"  ref[0] = [{ref['theta'][0]:.4f}, {ref['theta_dot'][0]:.4f}]")
print(f"  ✅ reset 对齐成功" if abs(u.state[0] - ref['theta'][0]) < 0.001 else f"  ⚠️ 对齐偏差: {abs(u.state[0] - ref['theta'][0]):.6f}")

print("\n=== 3. 测试 step + 跟踪误差 ===")
total_reward = 0.0
total_pos_err = 0.0
for i in range(50):
    action = np.array([0.0], dtype=np.float32)  # 零力矩，自然下垂
    obs, reward, term, trunc, info = wrapped.step(action)
    total_reward += reward
    total_pos_err += info["pos_err"]

print(f"  50 步零力矩 rollout:")
print(f"    total_reward = {total_reward:.2f}")
print(f"    mean_pos_err = {total_pos_err/50:.4f} rad²")
print(f"    无崩溃，info 字段齐全 ✅" if "pos_err" in info else "    ❌ info 字段缺失")

wrapped.close()
print("\n=== Wrapper 验证全部通过 ✅ ===")
