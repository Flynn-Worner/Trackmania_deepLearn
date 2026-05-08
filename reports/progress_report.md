# Trackmania Nations Forever Deep Learning Bot - Progress Report

## Project Overview
This project aims to create an autonomous driving agent for Trackmania Nations Forever (TMNF) using Reinforcement Learning, specifically Proximal Policy Optimization (PPO). Currently, the agent will rely on non-visual telemetry/memory data to minimize training time and complexity, avoiding the overhead of visual processing (CNNs).

## Project Architecture
The project is structured into the following key components:

1. **`interfacing/`**: Handles all interaction between the Python environment and the TMNF game.
   - `game_env.py`: Gym-like environment wrapper.
   - `memory_reader.py`: Retrieves real-time non-visual state data (speed, position, gear, etc.) from the game's memory.
2. **`data_processing/`**: Manages the preprocessing of data received from the interfacing layer.
   - `features.py`: Normalizes, scales, and prepares telemetry data as the state space input for the PPO agent.
3. **`models/`**: Contains the Reinforcement Learning model architectures.
   - `ppo_agent.py`: Implementation of the PPO algorithm (Actor-Critic networks).
4. **`training/`**: Orchestrates the RL training loop.
   - `train.py`: Handles the interaction between the `game_env` and `ppo_agent`, managing episodes, rewards, and model updates.
5. **`utils/`**: Helper scripts, configurations, and logging.
   - `config.py`: Hyperparameters and configuration constants.
   - `logger.py`: Tracks training metrics, rewards, and debugging info.

## Progress Log

### Phase 1: Setup and Architecture
- [x] Initialized Git repository.
- [x] Created project folder structure (`interfacing`, `data_processing`, `models`, `training`, `utils`, `reports`).
- [x] Established non-visual data strategy to expedite training.
- [x] Define the specific memory addresses/telemetry variables to extract.
- [x] Implement `memory_reader.py` to successfully hook into the TMNF process via Socket Plugins.

### Phase 2: Environment and Data
- [ ] Develop the Gym-compatible environment (`game_env.py`).
- [ ] Define Action Space (throttle, brake, steering).
- [ ] Define State Space (processed memory data).
- [ ] Implement Reward Shaping function (e.g., reward for forward progress, penalty for crashing/wall hits).
- [ ] Complete `data_processing/features.py` to standardize inputs.

### Phase 3: PPO Agent Implementation
- [ ] Set up Actor and Critic neural networks.
- [ ] Implement PPO loss functions and update rules.
- [ ] Configure hyperparameters in `config.py`.

### Phase 4: Training and Evaluation
- [ ] Run initial testing of the agent making random actions to ensure stability.
- [ ] Start preliminary training loop.
- [ ] Monitor logs and adjust reward shaping / hyperparameters.
- [ ] Train a converging model on a simple track.
