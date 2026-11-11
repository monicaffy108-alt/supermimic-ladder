"""L0 评估：两个口径衡量跟踪质量。

协议 A（起点对齐）：把杆子放在参考轨迹起点再跑，衡量纯跟踪精度。
协议 B（随机初始态）：含「从任意姿态刹车归位」，衡量恢复能力。

用法：
    cd supermimic_l0
    python eval_tracking.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import tracking_env as te

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = 'models/l0_ppo_curriculum.zip'
FIG_PATH = 'assets/l0_tracking_final.png'


def run_episode(env, model, aligned):
    if aligned:
        env.reset()                      # 先 reset，完成 Gymnasium 的内部记账
        env.t = 0.0
        env.unwrapped.state = np.array([te.theta_ref(0.0), te.dtheta_ref(0.0)],
                                       dtype=np.float64)
        obs = env._aug(env.unwrapped._get_obs())
    else:
        obs, _ = env.reset()

    rec = {'t': [], 'err': [], 'th': [], 'ref': [], 'tau': []}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        rec['t'].append(info['t'])
        rec['err'].append(info['err'])
        rec['th'].append(info['theta'])
        rec['ref'].append(info['ref'])
        rec['tau'].append(float(np.asarray(action).ravel()[0]))
    return {k: np.array(v) for k, v in rec.items()}


def main():
    te.set_amp(te.AMP_FINAL)
    model = PPO.load(MODEL_PATH, device='cpu')
    env = te.make_env()

    aligned = [run_episode(env, model, True) for _ in range(5)]
    random_ = [run_episode(env, model, False) for _ in range(5)]

    rmse_a = [float(np.sqrt(np.mean(r['err'] ** 2))) for r in aligned]
    mae_a = [float(np.mean(np.abs(r['err']))) for r in aligned]
    max_a = [float(np.max(np.abs(r['err']))) for r in aligned]
    rmse_b = [float(np.sqrt(np.mean(r['err'] ** 2))) for r in random_]

    print('=' * 52)
    print('协议 A：起点对齐参考轨迹（跟踪精度）')
    print(f'  RMSE    : {np.mean(rmse_a):.4f} ± {np.std(rmse_a):.4f} rad'
          f'  ({np.degrees(np.mean(rmse_a)):.2f}°)')
    print(f'  MAE     : {np.mean(mae_a):.4f} rad')
    print(f'  最大误差 : {np.mean(max_a):.4f} rad')
    print('协议 B：随机初始态（含刹车归位）')
    print(f'  RMSE    : {np.mean(rmse_b):.4f} ± {np.std(rmse_b):.4f} rad'
          f'  ({np.degrees(np.mean(rmse_b)):.2f}°)')
    print('=' * 52)

    best = aligned[int(np.argmin(rmse_a))]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'L0 跟踪结果（起点对齐）  RMSE={np.min(rmse_a):.3f} rad '
                 f'({np.degrees(np.min(rmse_a)):.1f}°)')
    ax[0].plot(best['t'], best['ref'], 'k--', lw=1.8, label='参考 θ_ref')
    ax[0].plot(best['t'], best['th'], 'b-', lw=1.4, label='实际 θ')
    ax[0].set_ylabel('角度 (rad)')
    ax[0].legend(loc='upper right')
    ax[0].grid(alpha=.3)
    ax[1].plot(best['t'], best['err'], 'r-', lw=1.2)
    ax[1].fill_between(best['t'], best['err'], 0, color='r', alpha=.12)
    ax[1].axhline(0, color='k', lw=.8)
    ax[1].set_ylabel('err (rad)')
    ax[1].grid(alpha=.3)
    ax[2].plot(best['t'], best['tau'], 'g-', lw=1.2)
    ax[2].axhline(te.MAX_TORQUE, color='r', ls='--', lw=.8)
    ax[2].axhline(-te.MAX_TORQUE, color='r', ls='--', lw=.8)
    ax[2].set_ylabel('力矩 τ')
    ax[2].set_xlabel('时间 t (秒)')
    ax[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=120, bbox_inches='tight')
    print(f'已保存: {FIG_PATH}')


if __name__ == '__main__':
    main()
