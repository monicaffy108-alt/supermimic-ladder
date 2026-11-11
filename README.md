# Supermimic Ladder

A step-by-step reproduction of **Supermimic** ([arXiv:2610.28480](https://arxiv.org/abs/2610.28480)): from a minimal sanity check in a toy environment, up to the full three-stage training loop.

Each rung is independently runnable and independently verifiable, so the project never starts by debugging a large physics simulator and a learned world model at the same time.

## Roadmap

| Level | Goal | Simulator / Algorithm | Status |
|---|---|---|---|
| **L0** | Minimal motion-imitation sanity check | Pendulum-v1 + PPO + hand-authored reference | ✅ Done |
| **L1** | Working PPO baseline | [MimicKit](https://github.com/xbpeng/MimicKit) + PPO | ⬜ Planned |
| **L2** | Swap PPO for the three-stage loop | Rollout → fit world model f_ω → BPTT update π_θ | ⬜ Planned |

## L0 Results

400k PPO steps with a curriculum-scheduled reference amplitude (≈3 min on CPU), tracking `θ_ref(t) = 0.3·sin(2π·0.2·t)`:

| Protocol | Setup | RMSE | MAE | Max error |
|---|---|---|---|---|
| **A — aligned start** | Pole initialised to the reference's starting state; measures pure tracking accuracy | **0.0805 rad (4.61°)** | 0.0704 rad | 0.1406 rad |
| **B — random start** | Gymnasium's random initial state, includes braking and recovery | 0.6277 ± 0.3234 rad (36.0°) | — | — |

![L0 tracking result](supermimic_l0/assets/l0_tracking_final.png)

Top: angle tracking (dashed black = reference, solid blue = actual). Middle: per-step tracking error. Bottom: applied torque.

Two observations worth keeping:

- The learned amplitude is 0.19–0.20 against a reference of 0.30 — **about 35% under-swing**, with a slight phase lag. This is the classic conservative-controller behaviour of PPO: the error is a squared penalty, overshoot accumulates more risk than undershoot, so the policy prefers to move less than to overshoot.
- Protocol B is far worse than A: **tracking is learned, recovery is not.** From a random initial pose the policy spends the first second saturating the torque limit just to brake. This is a common weakness when such policies move to real hardware.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# PyTorch — the CPU build is enough (an MLP policy is slower on GPU)
pip install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu

pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch, stable_baselines3, gymnasium; print(torch.__version__, stable_baselines3.__version__, gymnasium.__version__)"
```

## Usage

```bash
cd supermimic_l0

python train_curriculum.py    # trains and writes models/l0_ppo_curriculum.zip
python eval_tracking.py       # evaluates and writes assets/l0_tracking_final.png
```

`models/l0_ppo_curriculum.zip` ships with the repository, so you can reproduce the numbers without retraining — just run `eval_tracking.py`.

## Layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── supermimic_l0/
    ├── tracking_env.py         # reference phase in the observation, adjustable amplitude,
    │                           # and a torque-feasibility check
    ├── train_curriculum.py     # curriculum-scheduled amplitude training
    ├── eval_tracking.py        # two-protocol evaluation and plotting
    ├── assets/
    │   └── l0_tracking_final.png
    └── models/
        └── l0_ppo_curriculum.zip
```

## Three Pitfalls in L0

This section is worth more than the numbers. All three problems looked like "the training just didn't work", and none of them were.

**Pitfall 1 — the reference phase was not in the observation**

The observation initially reused Pendulum's native `[cosθ, sinθ, θ̇]`. But the target angle is a function of time: at `t = 1.25 s` the target is `+0.6`, at `t = 3.75 s` it is `−0.6`, and the two observations can be identical. The policy faced a partially observable problem — one input mapping to two opposite answers — so the gradient averaged them out and converged to *spinning at constant speed*, the one behaviour that does not need to distinguish phase.

Fix: append `[θ_ref, θ̇_ref]` to the observation, giving 5 dimensions. This is standard practice in motion imitation: the reference phase is an **input** to the policy, not a hidden variable.

**Pitfall 2 — the reference trajectory was infeasible in torque**

Pendulum-v1's dynamics are `θ̈ = 15·sinθ + 3u` — **undamped**; the damping coefficient appears only in the native reward. Holding the pole statically at angle θ requires `u = 5·sinθ`, and with `|u| ≤ 2` the largest statically holdable angle is `0.41 rad`.

The original reference amplitude was `1.0`, which requires a peak torque of 3.14 — 57% above the limit. The policy was **chasing a target it physically could not reach**: torque pinned at ±2, the pole spun faster and faster. Dropping the amplitude to 0.3 brings the requirement to 1.64, back inside the feasible region.

`tracking_env.required_peak_torque()` verifies this in one line before training starts.

**Pitfall 3 — doing nothing was a stronger local optimum**

At amplitude 0.3 the policy stopped spinning and instead pinned the pole at `θ = 0`. Parking there costs only `A²/2 = 0.045` per step, which is already cheap; and getting from "parked" to "tracking" requires passing through a phase where the pole moves and the error temporarily *grows*. The gradient only sees the immediate return, so the policy never leaves.

Fix: curriculum learning — the amplitude ramps from 0.05 to 0.3 over the first 250k steps. The policy stays inside the tracking basin throughout.

## Two Fundamental Design Choices

**Angular error must be wrapped**

`θ = π` and `θ = −π` are the same physical position; subtracting them directly yields a spurious `2π` error. Every angular difference is wrapped into `[−π, π]` with `atan2(sin(Δ), cos(Δ))`.

**The tracking reward fully overrides the native reward**

Pendulum-v1's native reward encourages the pole to sit still at the top, which directly conflicts with "keep swinging along a reference". The wrapper therefore discards the native reward entirely and uses only:

```
r = −(err² + 0.001·‖a‖²)
```

An earlier attempt that summed the native reward with a tracking penalty produced an agent that simply parked at the equilibrium.

## Citation

```bibtex
@article{supermimic2026,
  title   = {Supermimic},
  journal = {arXiv preprint arXiv:2610.28480},
  year    = {2026}
}
```
