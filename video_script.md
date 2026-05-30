# Trackmania AI Bot - Video Script

**Title Idea:** Teaching AI to Drive in Trackmania Nations Forever

---

### [Intro]
**Visual:** Quick montage of the car crashing, then a clip of it driving smoothly through a gate.
**Audio (Voiceover):** "Have you ever wondered how an AI learns to drive? Today, I'm breaking down how I built an autonomous bot to race in Trackmania Nations Forever. We'll look at how it sees the track, the brain that drives it, and the settings that make it all work."

---

### [Segment 1: The Bot's Structure]
**Visual:** On-screen diagram showing the TMInterface, the Python Environment, and the PPO Agent.
**Audio:** "The bot connects to the game using a tool called TMInterface, which acts as the bridge between Trackmania and our Python code. We use a custom Gymnasium environment that defines the track as a series of invisible gates. The AI's goal is simple: cross the gates in sequential order, as fast as possible, without crashing or getting stuck. 

For actions, it outputs three continuous values between -1 and 1: steering, gas, and brake. The steering is mapped smoothly to the game's inputs, while gas and brake act as binary triggers."

---

### [Segment 2: What the AI Sees (The 14 Observables)]
**Visual:** Gameplay footage with a futuristic overlay showing a live data array of 14 numbers fluctuating as the car drives.
**Audio:** "To drive effectively, the AI needs to 'see' its current state. At every single step, we feed it a 14-point observation array. All of these values are normalized so the neural network can process them easily:

1. **Speed:** The car's current speed.
2. **Yaw:** The car's rotation on the track.
3. **Pitch:** How much the car is tilting up or down.
4. **Roll:** How much the car is leaning side-to-side.
5. **Local Velocity X:** The car's lateral movement (sliding sideways).
6. **Local Velocity Y:** The car's vertical movement.
7. **Local Velocity Z:** The car's forward movement.
8. **Gate Vector X:** The horizontal direction to the center of the next target gate.
9. **Gate Vector Z:** The forward direction to the next target gate.
10. **Gate Distance:** The total distance to the next gate.
11. **Gate Progress:** The fraction of the track completed so far.
12. **Previous Steering:** The last steering action the AI took.
13. **Previous Gas:** The last acceleration action.
14. **Previous Brake:** The last braking action."

---

### [Segment 3: The Brain (Hyperparameters)]
**Visual:** Display of the PPO code snippet highlighting the hyperparameter values, maybe zooming in on specific numbers.
**Audio:** "The 'brain' behind the wheel is a Reinforcement Learning algorithm called PPO, or Proximal Policy Optimization. We built it with a neural network containing two hidden layers of 256 neurons each for both its decision-making policy and its value estimation.

Here are the key hyperparameters we used to train it:
- **Learning Rate (0.0003):** Kept small and stable to handle the noisy, chaotic nature of the game's physics.
- **N-Steps (2048):** The AI collects 2048 steps of experience across the environment before updating its brain.
- **Batch Size (256):** It learns from this gathered experience in chunks of 256.
- **Epochs (10):** It loops over its collected data 10 times per update to squeeze out as much learning as possible.
- **Gamma (0.99):** This discount factor tells the AI to care about long-term rewards, not just immediate speed.
- **Entropy Coefficient (0.01):** This adds a tiny bit of randomness, which encourages the AI to explore new driving lines early on instead of getting stuck doing the same thing."

---

### [Outro]
**Visual:** The AI successfully completing a lap with a smooth racing line. Call to action on screen (Like, Subscribe, GitHub link).
**Audio:** "Training this bot took a lot of trial and error, but watching it finally nail the perfect racing line makes it all worth it. If you want to check out the code or try training it yourself, the link is in the description. Thanks for watching!"
