import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from interfacing.env import TrackmaniaEnv


# ---------------------------------------------------------------------------
# Custom callback: logs episode metrics to TensorBoard frequently so you
# don't have to wait for a full n_steps rollout before seeing any data.
# ---------------------------------------------------------------------------

class TrainingMetricsCallback(BaseCallback):
    """
    Accumulates per-episode reward/length across all parallel envs and writes
    them to TensorBoard every `log_freq` env steps.  This means TB updates
    arrive every few minutes in debug mode rather than after a full rollout.
    """

    def __init__(self, log_freq: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_rew_buf = []
        self._ep_len_buf = []
        self._cur_rew = {}
        self._cur_len = {}

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [{}] * len(rewards))

        for idx, (r, d, info) in enumerate(zip(rewards, dones, infos)):
            self._cur_rew[idx] = self._cur_rew.get(idx, 0.0) + float(r)
            self._cur_len[idx] = self._cur_len.get(idx, 0) + 1
            if d:
                self._ep_rew_buf.append(self._cur_rew[idx])
                self._ep_len_buf.append(self._cur_len[idx])
                self._cur_rew[idx] = 0.0
                self._cur_len[idx] = 0

        if self.n_calls % self.log_freq == 0 and self._ep_rew_buf:
            window = self._ep_rew_buf[-20:]
            self.logger.record("custom/mean_ep_reward_20", float(np.mean(window)))
            self.logger.record("custom/mean_ep_length_20", float(np.mean(self._ep_len_buf[-20:])))
            self.logger.record("custom/n_episodes", len(self._ep_rew_buf))
            self.logger.dump(self.num_timesteps)

        return True


class AutoSaveCallback(BaseCallback):
    """
    Saves the model and VecNormalize stats every `save_freq` episodes.
    Keeps both a rolling 'latest' checkpoint and periodic numbered snapshots.
    """

    def __init__(self, save_freq_episodes: int = 50, model_path: str = "",
                 verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq_episodes
        self.model_path = model_path
        self._ep_count = 0
        self._last_save_ep = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        self._ep_count += int(sum(dones))

        if self._ep_count - self._last_save_ep >= self.save_freq:
            self._last_save_ep = self._ep_count
            self._save_checkpoint()
        return True

    def _save_checkpoint(self):
        # Save 'latest' (always overwritten)
        self.model.save(self.model_path)
        # Save VecNormalize stats if available
        env = self.model.get_env()
        if hasattr(env, "save"):
            env.save(self.model_path + "_vecnorm.pkl")

        # Also keep a numbered snapshot every 200 episodes
        if self._ep_count % 200 == 0:
            snap = f"{self.model_path}_ep{self._ep_count}"
            self.model.save(snap)
            if hasattr(env, "save"):
                env.save(snap + "_vecnorm.pkl")
            if self.verbose:
                print(f"\n💾 Snapshot saved → {snap}.zip (ep {self._ep_count})")

        if self.verbose:
            print(f"\n💾 Auto-save → {self.model_path}.zip (ep {self._ep_count})")


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(port: int):
    def _init():
        return TrackmaniaEnv(port=port)
    return _init


# ---------------------------------------------------------------------------
# Main training function – called by main.py or directly
# ---------------------------------------------------------------------------

def run_training(
    ports=None,
    total_timesteps: int = 500_000,
    tensorboard_dir: str = "./tensorboard/",
    model_path: str = "models/saved/ppo_trackmania_final",
    debug: bool = False,
    force_new: bool = False,
):
    """
    Train the PPO agent.

    Parameters
    ----------
    ports : list[int]
        One TMInterface port per running TM window.  Defaults to [8483].
    total_timesteps : int
        How many env steps to train for.
    tensorboard_dir : str
        Absolute path is strongly recommended; relative paths resolve from cwd.
    model_path : str
        Save/load path without .zip extension.
    debug : bool
        Use smaller n_steps (128) so TensorBoard updates are visible within
        a couple of minutes.  Switch off for production runs.
    force_new : bool
        Ignore any existing checkpoint and create a fresh model.
    """
    if ports is None:
        ports = [8483]

    # Hyper-parameters: debug mode uses small n_steps so you see TB data fast.
    n_steps = 128 if debug else 2048
    batch_size = 64 if debug else 256
    log_freq = 50 if debug else 200

    print(f"Initializing {len(ports)} environment(s) on port(s) {ports} ...")
    raw_env = SubprocVecEnv([make_env(p) for p in ports])

    # VecNormalize: online running mean/std for observations AND rewards.
    # This is critical for continuous-action PPO stability.
    # Stats are saved alongside the model so inference can reproduce them.
    env = VecNormalize(
        raw_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=20.0,
    )

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    vecnorm_path = model_path + "_vecnorm.pkl"

    if not force_new and os.path.exists(model_path + ".zip"):
        print(f"Loading existing model from {model_path}.zip ...")
        model = PPO.load(model_path, env=env, tensorboard_log=tensorboard_dir)
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, raw_env)
            env.training = True
            model.set_env(env)
            print(f"VecNormalize stats loaded from {vecnorm_path}")
    else:
        print("Creating new PPO model ...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            # Learning rate: small but stable for noisy game envs.
            learning_rate=0.0003,
            # n_steps: collect this many steps per env before each update.
            # 128 in debug makes TB feel alive; 2048 is standard production.
            n_steps=n_steps,
            batch_size=batch_size,
            # More epochs per update improves sample efficiency.
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            # Entropy coefficient: keeps exploration alive early on.
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            # Wider networks handle the nonlinear reward landscape better.
            policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
            tensorboard_log=tensorboard_dir,
        )

    metrics_cb = TrainingMetricsCallback(log_freq=log_freq)
    autosave_cb = AutoSaveCallback(
        save_freq_episodes=50,
        model_path=model_path,
    )
    callback = CallbackList([metrics_cb, autosave_cb])

    print()
    print("=" * 56)
    print("  TRAINING STARTED  (Ctrl+C to stop and save)")
    print(f"  n_steps per env : {n_steps}")
    print(f"  batch_size      : {batch_size}")
    print(f"  Auto-save every : 50 episodes")
    print(f"  TensorBoard dir : {tensorboard_dir}")
    print("=" * 56)
    print()
    print("Steering note: the policy outputs a continuous value in [-1, 1]")
    print("which is linearly mapped to TM's integer steer range [-65536, 65536].")
    print("Gas and brake remain binary (game API limitation).")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name="PPO_Training",
            reset_num_timesteps=False,
            callback=callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        model.save(model_path)
        env.save(vecnorm_path)
        print(f"\nModel saved        → {model_path}.zip")
        print(f"VecNormalize stats → {vecnorm_path}")
        # env.close() blocks on remote.recv() waiting for worker confirmation.
        # A second Ctrl+C during this wait raises another KeyboardInterrupt and
        # produces an ugly traceback.  Suppress it — the game recovers on its own
        # within ~2 s when its socket timeout fires.
        try:
            env.close()
        except (KeyboardInterrupt, Exception):
            pass


# ---------------------------------------------------------------------------
# Backward-compatible direct execution: python training/train.py
# ---------------------------------------------------------------------------

def main():
    """Direct execution entry point (single env, default port 8483)."""
    run_training()


if __name__ == "__main__":
    main()
