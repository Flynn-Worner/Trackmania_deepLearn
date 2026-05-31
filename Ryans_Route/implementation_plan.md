# Implementation Plan: Replay-Driven Gate Reward System

## 1. Context & Objective
**Goal:** Pivot from GBX-block spatial rewards to a sequential, gate-based reward system. 
**Method:** 1. Human drives the ideal line; system records positional/rotational waypoints.
2. Waypoints are expanded into 2D/3D "gates" spanning track width.
3. RL Agent is rewarded exclusively for intersecting these gates sequentially as fast as possible.
**Constraint:** Preserve `SubprocVecEnv` multi-instancing for training; strictly separate the recording tool from the training loop.

## 2. Phase 1: Waypoint Recorder (`record_waypoints.py`)
Create a new standalone script to capture the human racing line.
* **Connection:** Connect via `TMInterface` (single instance, port 8483).
* **Trigger:** Start recording when `simManager.InRace` becomes true.
* **Capture Logic:** Read state every `N` ticks (recommend every 0.1s to 0.2s of race time, or based on a fixed distance threshold, e.g., every 10 meters).
* **Data Payload (Per Waypoint):**
    * `time`: Race time (ms)
    * `position`: [x, y, z]
    * `rotation`: [pitch, yaw, roll] (Yaw is critical for gate orientation)
* **Output:** Save to `data/waypoints.json`.

## 3. Phase 2: Gate Generator (`generate_gates.py`)
Create a processing script to convert `waypoints.json` into `gates.json`.
* **Math/Geometry:**
    * For each waypoint, isolate the `yaw` (horizontal rotation).
    * Calculate the perpendicular vector to the car's forward heading.
    * Project this vector outward by `TRACK_WIDTH / 2` in both directions to establish `left_post` and `right_post` [x, y, z] coordinates.
* **Data Structure (`gates.json`):**
    ```json
    [
      {
        "gate_id": 0,
        "center": [x, y, z],
        "left_post": [x, y, z],
        "right_post": [x, y, z],
        "target_speed": 250.0 // Optional: derived from human record for shaping
      }
    ]
    ```
* **Deprecation:** `extract_spline.py` and `data/map_blocks.json` are now fully obsolete and can be deleted.

## 4. Phase 3: Environment & Reward Overhaul (`env.py` & `rewards.py`)
Rewrite the `step()` and reward logic to consume `gates.json`.
* **State Tracking (in `env.py`):**
    * Maintain `current_target_gate_idx` for each episode.
    * On reset, set to `0`.
* **Intersection Math:**
    * Treat the car's movement between `t-1` and `t` as a line segment.
    * Treat the target gate (`left_post` to `right_post`) as a line segment.
    * Use 2D segment intersection (ignoring Y/height initially, or checking if Y is within a reasonable tolerance) to register a gate pass.
* **Reward Rules (`rewards.py`):**
    * *Primary:* Massive dense reward for intersecting `current_target_gate_idx`. Increment index.
    * *Shaping (Optional):* Small dense reward for closing the distance to `current_target_gate_idx`'s center.
    * *Penalties:* Standard step penalty (to encourage speed).
    * *Terminations:* * `STUCK`: Fired if speed drops below `V_min` after grace period.
        * `WRONG_WAY`: Fired if distance to `target_gate` increases beyond a threshold.
        * `FINISH`: Reached the final gate array index.

## 5. Phase 4: Multi-Instancing Integrity
* Ensure `gates.json` is loaded into memory **once** on initialization by the main process, or cleanly by each worker without I/O blocking.
* Retain the existing `train.py` argument structure (`--ports`, `--new`).
* Ensure the `Python_Link.as` plugin remains untouched; it only passes I/O, which the new `env.py` will handle normally.