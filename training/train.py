import os
import sys

# Ensure root is in path so we can import interfacing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfacing.game_env import TrackmaniaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(port):
    """Utility function to create an environment for a specific port."""
    def _init():
        return TrackmaniaEnv(port=port)
    return _init

def main():
    # LIST OF PORTS: You must open a Trackmania window and TMInterface for EACH of these ports.
    # We are reverting back to 1 instance to avoid connection issues.
    PORTS = [8483]
    
    print(f"Initializing {len(PORTS)} parallel environments...")
    
    # Wrap multiple environments in parallel processes
    env = SubprocVecEnv([make_env(p) for p in PORTS])
    
    print("Skipping check_env as it can crash synchronous game environments...")
    
    save_path = "models/saved/ppo_trackmania_final"
    
    if os.path.exists(save_path + ".zip"):
        print(f"\nLoading existing model from {save_path}.zip to resume training...")
        model = PPO.load(save_path, env=env, tensorboard_log="./tensorboard/")
    else:
        print("\nCreating new PPO Agent...")
        # n_steps: How many frames to collect before updating the brain (2048 is standard)
        # ent_coef: High entropy forces the AI to explore randomly instead of getting stuck!
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, 
            n_steps=2048, 
            batch_size=64,
            ent_coef=0.01,
            tensorboard_log="./tensorboard/"
        )
    
    print("\n=======================================================")
    print("STARTING TRAINING! (Press Ctrl+C in terminal to stop)")
    print("=======================================================")
    print("Ensure TMInterface is running in TMNF and you are on a track.")
    
    try:
        # Train for a large number of timesteps! (It will run until you press Ctrl+C)
        model.learn(total_timesteps=500000, tb_log_name="PPO_Training", reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Always save the model
        os.makedirs("models/saved", exist_ok=True)
        model.save(save_path)
        print(f"\nModel saved to {save_path}.zip")
        env.close()

if __name__ == "__main__":
    main()
