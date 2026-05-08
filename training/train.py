import os
import sys

# Ensure root is in path so we can import interfacing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfacing.game_env import TrackmaniaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def main():
    print("Initializing environment...")
    
    # We use a lower ticks_per_step (10 = 10 updates a second) 
    env = TrackmaniaEnv(port=8483, ticks_per_step=10)
    
    print("Checking environment compatibility with Stable Baselines 3...")
    try:
        check_env(env)
        print("Environment passes Gymnasium compatibility checks!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        # We will continue anyway, sometimes check_env is too strict
        pass
    
    print("\nCreating PPO Agent...")
    # We use MlpPolicy since our observations are just 7 floats
    # n_steps is set relatively low for faster updates during testing
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=1024, 
        batch_size=64,
        tensorboard_log="./tensorboard/"
    )
    
    print("\n=======================================================")
    print("STARTING TRAINING! (Press Ctrl+C in terminal to stop)")
    print("=======================================================")
    print("Ensure TMInterface is running in TMNF and you are on a track.")
    
    try:
        # Train for a small number of timesteps as an initial test
        model.learn(total_timesteps=10000, tb_log_name="PPO_First_Test")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Always save the model
        os.makedirs("models/saved", exist_ok=True)
        save_path = "models/saved/ppo_trackmania_basic"
        model.save(save_path)
        print(f"\nModel saved to {save_path}.zip")
        env.close()

if __name__ == "__main__":
    main()
