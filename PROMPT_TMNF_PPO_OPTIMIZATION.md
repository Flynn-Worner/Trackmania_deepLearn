# Claude prompt key: TMNF PPO driving agent (optimization and fixes)

Copy everything below the line into a new Claude conversation (or attach this file). The coding agent should **read the referenced project files** and apply changes only where allowed.

---

## Your role

You are a senior ML engineer and gameplay-tools developer. Your job is to **improve learning stability, TensorBoard visibility, continuous control semantics, and multi-instance training** for a Trackmania Nations Forever (TMNF) PPO agent that talks to the game over **TMInterface-style sockets** (Python `TMInterface` class + Openplanet AngelScript `Python_Link.as`).

## Hard scope (files you may change)

Edit **only** these paths (create `main.py` if it does not exist):

| File | Purpose |
|------|---------|
| `training/train.py` | SB3 PPO, vectorized envs, logging, hyperparameters |
| `interfacing/game_env.py` | Gymnasium env, rewards, observations, message loop |
| `interfacing/Python_Link.as` | Socket server in TM; port and input handling |
| `extract_spline.py` | GBX → `data/map_blocks.json` (or related spline data) |
| `main.py` | CLI entry: repo-root cwd, ports, timesteps, tensorboard dir |

**Do not** refactor unrelated modules, rename the whole project, or add unrelated dependencies without explicit user approval. Match existing naming, imports, and comment style.

## Mandatory pre-read (current architecture)

### Data flow

1. **Training**: `SubprocVecEnv` spawns one process per environment. Each process constructs `TrackmaniaEnv(port=...)`.
2. **Connection**: Each env connects to `127.0.0.1:port` via `TMInterface.register()`. The game-side plugin (`Python_Link.as`) **listens** on that port and exchanges binary messages (`MessageType` enum aligned between Python and AngelScript).
3. **Step sync**: `set_on_step_period(ticks_per_step)` controls how many simulation ticks occur between `SC_RUN_STEP_SYNC` messages. The env’s `step()` blocks in a `while True` loop reading message types until `SC_RUN_STEP_SYNC`, applying rewards and observations, and responding with `_respond_to_call`.
4. **Actions**: `game_env` uses `gymnasium.spaces.Box(low=-1, high=1, shape=(3,))` for `[steer, gas, brake]`. Steering is mapped to `int` in `[-65536, 65536]` for `set_input_state`. Gas and brake are **thresholded** to booleans (`> 0`).
5. **Observations**: Six floats: speed, min 2D distance to block centers (in `_get_observation`), yaw, pitch, roll, checkpoint count; passed through `StateNormalizer` in `data_processing/features.py` (clipping only; **you may not change `features.py` unless the user expands scope**—if normalization must change, do it inside `game_env.py` with clear comments or ask to widen scope).

### Snapshot of current behaviors (ground truth for diffs)

**`training/train.py`**

- Builds `SubprocVecEnv([make_env(p) for p in PORTS])` with `PORTS = [8483]` (single instance commented as intentional).
- `PPO("MlpPolicy", env, ... n_steps=2048, batch_size=64, ent_coef=0.01, tensorboard_log="./tensorboard/")`.
- Loads from `models/saved/ppo_trackmania_final.zip` if present; `learn(total_timesteps=500000, tb_log_name="PPO_Training", reset_num_timesteps=False)`.

**`interfacing/game_env.py`**

- Loads `data/map_blocks.json`, optionally reorders blocks into a greedy spatial “snake” from a block whose name contains `start`.
- Rewards: strong speed² term, block-index progress jumps, small distance penalty, per-step time penalty, OOB Y threshold, large speed-drop “crash” termination, stuck-at-low-speed termination, checkpoint/finish bonuses in checkpoint handler.
- **Risk**: In `step()`, `reward` is set to `0.0` once at the top, then inside the message loop `SC_RUN_STEP_SYNC` assigns `reward = (speed_factor ** 2) * 2.0` from scratch. If `SC_CHECKPOINT_COUNT_CHANGED_SYNC` is processed **in the same Python `step()` wait** before the next `SC_RUN_STEP_SYNC`, the checkpoint bonus (`reward += 50.0`) can be **wiped** when the run-step branch runs. **Fix by accumulating rewards** across all messages that belong to the same env step, or by a clearly documented message order guarantee from the plugin.

**`interfacing/Python_Link.as`**

- `Main()` sets `PORT = 8483` and listens. **This blocks multi-instance training** unless each TM process uses a different port (separate plugin build, configurable port, or duplicated plugin folder with edited port).
- `CSetInputState`: reads accel/brake as uint8 and steer as int32; maps steer to left/right discrete input magnitudes per TM API.

**`extract_spline.py`**

- Reads `data/map.gbx` via pygbx, filters block names, writes `data/map_blocks.json` with grid and `world_center` coordinates.

**`main.py`**

- **Missing** in the baseline repo: add a small entrypoint that `chdir`s to repo root (or documents required cwd), parses CLI args or env vars for `--ports`, `--timesteps`, `--tensorboard`, `--model-path`, and invokes training so TensorBoard paths are never ambiguous.

## Reference project: `reference_linesight/`

The user keeps a similar-scope project in **`reference_linesight/`** (may be **gitignored** in this repo, so it might not appear in zip exports or sandboxes).

**Before changing algorithms or env structure**, you must:

1. Open `reference_linesight/` on the **same machine** as this project (or ask the user to paste tree + key files).
2. Identify and briefly note (in your reply to the user, not necessarily in code comments): how it configures **ports**, **vectorized envs**, **reward shaping**, **observation normalization**, **PPO hyperparameters**, and **TensorBoard / callbacks**.
3. Port **patterns**, not filenames blindly: align with this repo’s `TMInterface` / `TrackmaniaEnv` API.

If `reference_linesight` is unavailable, state that explicitly and proceed using RL best practices below—but still flag unknowns (e.g. analog inputs).

## Goals (priority order)

1. **TensorBoard actually updates on a human-relevant timescale**  
   - Remember: SB3 logs many scalars **after each rollout**; with `n_steps=2048` and a **slow real-time env**, the first event can take a very long time.  
   - Implement: debug mode with smaller `n_steps`, **custom Callback** logging env-step counters / episodic return / speed / terminal flags on a fixed interval, and **absolute** tensorboard log directory rooted at the project root (via `main.py`).  
   - Document the exact `tensorboard --logdir ...` command for the user.

2. **Multi-instance Trackmania training**  
   - Each `SubprocVecEnv` worker must connect to a **unique TCP port**.  
   - Change `Python_Link.as` (or document duplicate plugins) so each TMNF instance listens on its assigned port.  
   - Update `train.py` / `main.py` to accept a list of ports matching the number of launched games.  
   - Document startup: N TM windows → N plugins / N ports → load same map → run `python main.py ...`.

3. **Continuous steering (and clarify “continuous”)**  
   - The policy already outputs continuous `steer ∈ [-1, 1]`; the game layer uses **integer** steer magnitude. Preserve smooth mapping (e.g. clip, optional deadzone, document scaling).  
   - If the user wants **continuous throttle/brake**, confirm whether `simManager.SetInputState` supports analog values beyond 0/1 in their TM/Openplanet version; only then extend the wire protocol (`CSetInputState`) and Python `set_input_state`. Otherwise document the limitation.

4. **Learning quality**  
   - Fix reward/message bugs (checkpoint accumulation).  
   - Rebalance or normalize reward components so the value function is not dominated by unbounded speed² without variance control.  
   - Consider `VecNormalize` (would require `stable_baselines3` imports and wrapping—**if you add this**, keep it in `train.py` only and document `venv` deps).  
   - Tune PPO for **slow** envs: learning rate schedule, `n_epochs`, `clip_range`, network width, `gamma`, `gae_lambda`.

5. **Spline / progress signal**  
   - Greedy block chaining can fail on loops or overlapping routes. Improve `extract_spline.py` or the consumption logic in `game_env.py` using ideas from `reference_linesight` (e.g. spline arc-length, checkpoint order, or CP-based progress).

## Dependencies

- Ensure `requirements.txt` lists everything `train.py` imports (e.g. **`stable-baselines3`**, **`torch`**, **`gymnasium`**, **`tminterface`**, **`numpy`**). Add versions only if the user already pins elsewhere; otherwise minimal addition is acceptable.

## Acceptance criteria

- [ ] Running training from repo root via `main.py`, TensorBoard shows **new scalars within a few minutes** using a documented “debug” configuration (even if production `n_steps` stays large).
- [ ] With **two ports and two TM instances**, both envs train without cross-connection (verify distinct PIDs / distinct ports in logs).
- [ ] Steering uses the full intended range without unnecessary binarization at the RL layer; integer mapping documented.
- [ ] Checkpoint rewards are not lost due to message-order overwrites within `step()`.
- [ ] User-facing **short** README snippet at end of your reply (not necessarily a new file) listing: command line for training, command line for TensorBoard, and multi-instance checklist.

## Testing protocol (you must follow mentally when implementing)

1. **Smoke**: one env, one port, short `total_timesteps`, verify connect, step, reset, TB file creation.
2. **Scale**: two envs, two ports, two games; confirm no socket mix-ups.
3. **Regression**: single-env training still works when only one port is passed.

## What not to do

- Do not edit `.gitignore` or the plan file unless the user asks.
- Do not add large binary assets.
- Do not remove the user’s ability to resume from `models/saved/ppo_trackmania_final` without documenting the new path/flag behavior.

---

## Optional: one-line user reminder

Keep `reference_linesight/` readable by your AI session (remove from `.gitignore` locally or paste critical files) so comparisons are grounded in real code.
