# Trackmania Deep Learning (PPO Agent) 🏎️🧠

## Project Overview
This project trains an autonomous racing agent to drive in *Trackmania Nations Forever* using Deep Reinforcement Learning (Proximal Policy Optimization). The bot learns to navigate custom tracks by interpreting its velocity, orientation, and distance to dynamically generated "gates" along the ideal racing line.

## Group Information
**Group Number:** 13
**Group Members:**
- Flynn Worner 
- Ryan Gleeson

---

## 🛠️ Architecture & Environment Setup

The agent interacts with the game engine via a custom Python Gym environment that communicates directly with the Trackmania client over a TCP socket using **TMInterface**.

### Environment Flow Diagram
```text
                       [Action Space: Steer, Gas, Brake]
                      +---------------------------------+
                      |                                 |
                      V                                 |
+---------------------------+                 +---------------------------+
|      PPO Neural Network   |                 | Trackmania Nations Forever|
|  (stable-baselines3)      |                 |       (TMInterface)       |
+---------------------------+                 +---------------------------+
                      |                                 |
                      |                                 |
                      +---------------------------------+
                       [Observation Space: 14 Floats]
                        Speed, Yaw, Pitch, Roll, Velocity
                        Gate dx/dz, Gate Distance, Progress
```

### The "Gate" System
Instead of blindly wandering, we record an ideal racing line as waypoints. These waypoints are converted into invisible perpendicular gates that the bot must cross in sequence.

```text
       Gate 1                 Gate 2                 Gate 3
  |-------------|        |-------------|        |-------------|
  |      X      |  --->  |      X      |  --->  |      X      |
  |             |        |             |        |             |
=================================================================== Track Wall
```

---

## ⚙️ Installation & Prerequisites

To run this project, you need the base game, the interface plugin, and the Python environment.

### 1. Install Trackmania Nations Forever & TMInterface
You cannot run this code without having the game running in the background.
1. **Trackmania Nations Forever (TMNF):** Download and install the game (available for free on Steam).
2. **TMInterface:** This is the tool that allows Python to read memory and send inputs to the game.
   - [TMInterface Official Website & Installation Guide](https://donadigo.com/tminterface/)
   - Follow the instructions to install the plugin and inject it into your TMNF client.

### 2. Install Python Dependencies
Ensure you have Python 3.9+ installed. Clone this repository and install the requirements:
```bash
git clone https://github.com/Flynn-Worner/Trackmania_deepLearn.git
cd Trackmania_deepLearn
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

Running the project consists of three main phases: Recording a line, generating the gates, and training the model.

### Phase 1: Record the Racing Line
Before training, you must teach the bot the layout of the track.
1. Open Trackmania and load your track. Ensure TMInterface is running on port `8483`.
2. Run the recorder script:
   ```bash
   python main.py --record
   ```
3. The game speed will slow down to 0.5x. Drive a clean lap down the middle of the track. The script will drop a waypoint every 10 meters and save them to `data/waypoints.json`.

### Phase 2: Generate Training Gates
Convert your driven waypoints into invisible gates that span the width of the track.
```bash
python generate_gates.py --width 32
```
This generates `data/gates.json`, which the environment uses for reward calculations.

### Phase 3: Train the Model
Start the PPO training loop. The script will automatically connect to the game and begin driving.
```bash
python main.py --ports 8483 --timesteps 5000000
```
- **To view live training graphs:** Open a new terminal and run `tensorboard --logdir tensorboard`.
- **To start fresh (wipe old model):** Add the `--new` flag.

### Expected Behavior
- When training begins, the car will spawn at the start line and take seemingly random, erratic actions.
- It will hit walls, stop moving, or go backwards. If it stops or drives backward for too long, the episode will instantly reset.
- As it crosses gates, it receives a reward of `+20` per gate. Over thousands of episodes, the neural network optimizes its steering and throttle to maximize this reward, eventually learning a smooth racing line to the finish.
