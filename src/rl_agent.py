"""
RL Load Scheduling Agent.

Environment:
  State  : [normalized_forecast, hour_sin, hour_cos, load_on (0/1)]
  Actions: 0 = defer load, 1 = run load
  Reward : cost_reduction_vs_baseline - comfort_penalty

Agent: Q-Learning with epsilon-greedy exploration (tabular, discretized state).

Output directories:
  experiments/results/   ← rl_agent.json (saved Q-table)
  experiments/logs/      ← rl_learning_curve.png (training log)
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = "experiments/results"   # saved agent (Q-table)
LOGS_DIR    = "experiments/logs"      # learning curve plot


# ── Environment ────────────────────────────────────────────────────────────────

class LoadSchedulingEnv:
    """
    Simulated single-device load scheduling environment.
    Episode = one full day (24 hourly steps).
    Reward  = cost reduction vs always-on baseline - comfort penalty.
    """

    MUST_RUN_HOURS = {7, 8, 18, 19, 20}

    def __init__(self, forecasts: np.ndarray, hours: np.ndarray,
                 cost_per_kwh: float = 1.0, comfort_penalty: float = 2.0,
                 min_run_hours: int = 4):
        self.forecasts       = forecasts
        self.hours           = hours
        self.cost_per_kwh    = cost_per_kwh
        self.comfort_penalty = comfort_penalty
        self.min_run_hours   = min_run_hours
        self.n_steps         = 24
        self._ptr            = 0
        self.reset()

    def reset(self):
        self.step_idx    = 0
        self.load_on     = 0
        self.runs_so_far = 0
        self._ptr = random.randint(0, max(0, len(self.forecasts) - self.n_steps - 1))
        return self._obs()

    def _obs(self):
        f     = float(self.forecasts[self._ptr + self.step_idx])
        h     = int(self.hours[self._ptr + self.step_idx]) % 24
        h_sin = np.sin(2 * np.pi * h / 24)
        h_cos = np.cos(2 * np.pi * h / 24)
        return np.array([f, h_sin, h_cos, float(self.load_on)], dtype=np.float32)

    def step(self, action: int):
        hour     = int(self.hours[self._ptr + self.step_idx]) % 24
        forecast = float(self.forecasts[self._ptr + self.step_idx])

        baseline_cost = forecast * self.cost_per_kwh
        agent_cost    = forecast * self.cost_per_kwh * action
        reward        = baseline_cost - agent_cost

        if hour in self.MUST_RUN_HOURS and action == 0:
            reward -= self.comfort_penalty

        self.load_on      = action
        self.runs_so_far += action
        self.step_idx    += 1

        done = self.step_idx >= self.n_steps
        if done and self.runs_so_far < self.min_run_hours:
            reward -= self.comfort_penalty * (self.min_run_hours - self.runs_so_far)

        next_obs = self._obs() if not done else np.zeros(4, dtype=np.float32)
        info     = {"hour": hour, "forecast": forecast,
                    "action": action, "reward": reward}
        return next_obs, reward, done, info


# ── State discretizer ──────────────────────────────────────────────────────────

class StateDiscretizer:
    def __init__(self, bins=10):
        self.bins  = bins
        self.lows  = np.array([-3.0, -1.0, -1.0, 0.0])
        self.highs = np.array([ 3.0,  1.0,  1.0, 1.0])

    def discretize(self, obs: np.ndarray) -> tuple:
        clipped        = np.clip(obs, self.lows, self.highs)
        scaled         = (clipped - self.lows) / (self.highs - self.lows)
        binned         = (scaled * (self.bins - 1)).astype(int)
        binned[-1]     = int(obs[-1])
        return tuple(binned)


# ── Q-Learning agent ──────────────────────────────────────────────────────────

class QLearningAgent:
    def __init__(self, n_actions=2, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, bins=10):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.disc          = StateDiscretizer(bins)
        self.Q             = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[self.disc.discretize(obs)]))

    def update(self, obs, action, reward, next_obs, done):
        s  = self.disc.discretize(obs)
        s2 = self.disc.discretize(next_obs)
        td = reward + (0 if done else self.gamma * np.max(self.Q[s2]))
        self.Q[s][action] += self.alpha * (td - self.Q[s][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {str(k): v.tolist() for k, v in self.Q.items()}
        with open(path, "w") as f:
            json.dump({"Q": data, "epsilon": self.epsilon}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        for k, v in data["Q"].items():
            self.Q[eval(k)] = np.array(v)
        self.epsilon = data["epsilon"]


# ── Training loop ──────────────────────────────────────────────────────────────

def train_rl_agent(
    forecasts: np.ndarray,
    hours: np.ndarray,
    n_episodes: int = 500,
    seed: int = 42,
    save_path: str = None,
    plot_path: str = None,
):
    """
    Train Q-Learning agent.

    Saves:
      Q-table JSON      → experiments/results/rl_agent.json
      Learning curve    → experiments/logs/rl_learning_curve.png
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "rl_agent.json")
    if plot_path is None:
        plot_path = os.path.join(LOGS_DIR, "rl_learning_curve.png")

    random.seed(seed)
    np.random.seed(seed)

    env   = LoadSchedulingEnv(forecasts, hours)
    agent = QLearningAgent()

    episode_rewards = []

    for ep in range(n_episodes):
        obs          = env.reset()
        total_reward = 0.0
        done         = False

        while not done:
            action          = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            total_reward   += reward
            obs             = next_obs

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            mean_r = np.mean(episode_rewards[-50:])
            print(f"[RL] Episode {ep+1:4d}/{n_episodes} | "
                  f"Mean Reward (last 50): {mean_r:.3f} | "
                  f"ε={agent.epsilon:.3f}")

    # ── Save Q-table → experiments/results/ ───────────────────────────────────
    agent.save(save_path)
    print(f"[RL] Agent saved → {save_path}")

    # ── Save learning curve → experiments/logs/ ───────────────────────────────
    _save_learning_curve(episode_rewards, plot_path)

    return agent, episode_rewards


def _save_learning_curve(rewards, path, window=20):
    """Save RL learning curve to experiments/logs/."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rewards  = np.array(rewards)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(range(window - 1, len(rewards)), smoothed,
            color="steelblue", linewidth=2, label=f"{window}-ep moving avg")
    ax.axhline(0, color="red", linestyle="--", linewidth=1,
               label="Baseline (always-on)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("RL Agent Learning Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[RL] Learning curve → {path}")


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_agent(agent, forecasts, hours, n_episodes=50, seed=99):
    random.seed(seed)
    np.random.seed(seed)
    env = LoadSchedulingEnv(forecasts, hours)
    agent_rewards, baseline_rewards = [], []

    for _ in range(n_episodes):
        obs  = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, r, done, _ = env.step(action)
            ep_r += r
        agent_rewards.append(ep_r)

        obs  = env.reset()
        ep_b = 0.0
        done = False
        while not done:
            _, r, done, _ = env.step(1)
            ep_b += r
        baseline_rewards.append(ep_b)

    print(f"[RL Eval] Agent    reward: {np.mean(agent_rewards):.3f} "
          f"± {np.std(agent_rewards):.3f}")
    print(f"[RL Eval] Baseline reward: {np.mean(baseline_rewards):.3f} "
          f"± {np.std(baseline_rewards):.3f}")
    return agent_rewards, baseline_rewards


if __name__ == "__main__":
    T              = 5000
    fake_forecasts = np.random.randn(T).astype(np.float32)
    fake_hours     = np.tile(np.arange(24), T // 24 + 1)[:T].astype(np.float32)
    agent, rewards = train_rl_agent(fake_forecasts, fake_hours, n_episodes=200)
    evaluate_agent(agent, fake_forecasts, fake_hours)
