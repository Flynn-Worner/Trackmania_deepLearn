# Multi-Instance Trackmania Training Guide

Running N TM windows in parallel means N environments collecting experience simultaneously — this is the biggest practical speed-up you can make to PPO training.

The core challenge is that every TM window loads plugins from the **same folder**, so by default every instance would try to listen on port 8483 and all but the first would fail.

This guide explains three methods, from easiest to most involved.

---

## Why ports matter

```
TM Window 1 ──► Python_Link.as (port 8483) ◄──► TrackmaniaEnv(port=8483)
TM Window 2 ──► Python_Link.as (port 8484) ◄──► TrackmaniaEnv(port=8484)
TM Window 3 ──► Python_Link.as (port 8485) ◄──► TrackmaniaEnv(port=8485)
                                                       │
                                               SubprocVecEnv
                                                       │
                                                  PPO.learn()
```

Each TM window must listen on its **own unique port**. Python connects to each port with a separate `TrackmaniaEnv` worker process.

---

## Method 1 — Auto-scan (recommended, no config needed)

The `Python_Link.as` plugin already has auto-scan built in. On startup it tries ports 8483, 8484, 8485 … 8492 in order and binds to the first free one.

**This means you only need to open TM windows in order — no extra configuration.**

### Step-by-step

1. **Copy the plugin** to your TMInterface Plugins folder (once):

   ```
   %USERPROFILE%\Documents\TMInterface\Plugins\Python_Link.as
   ```

   Copy `interfacing\Python_Link.as` from this repo there.

2. **Open TM instance 1** — let it fully load. Plugin binds port 8483.

3. **Open TM instance 2** — let it fully load. Plugin binds port 8484.

4. **Open TM instance N** — plugin binds port 8485, 8486, …

5. In **each** TM window, load your training map and get into a race.

6. Check the TMInterface log (F3 in TM) — you should see:
   ```
   Python Link: auto-bound to port 8483
   ```
   ```
   Python Link: auto-bound to port 8484
   ```

7. Run training:
   ```powershell
   python main.py --ports 8483 8484        # 2 instances
   python main.py --ports 8483 8484 8485   # 3 instances
   python main.py --ports 8483 8484 --debug  # debug mode
   ```

### Troubleshooting auto-scan

| Symptom | Fix |
|---------|-----|
| Both windows show port 8483 | Auto-scan try/catch may not work in your Openplanet build — use Method 2 or 3 |
| Plugin log shows "could not bind to any port" | Another app is using 8483–8492; close it or change the scan range in Python_Link.as |
| Python can't connect | Make sure you started training AFTER all TM windows loaded |

---

## Method 2 — Sequential launch script (fully automated)

`scripts\launch_multienv.ps1` automates the plugin-swap approach: it patches the plugin with the correct port, launches TM, waits for it to load, then moves to the next instance.

### Requirements

- Windows PowerShell 5+ (built into Windows 10/11)
- TmForever.exe accessible (auto-detected from common paths)
- TMInterface installed

### Usage

Open PowerShell in the repo root and run:

```powershell
# Allow running local scripts (only needed once per machine)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Launch 2 instances (ports 8483, 8484), then prompt you to load the map
.\scripts\launch_multienv.ps1

# Launch 3 instances
.\scripts\launch_multienv.ps1 -Ports 8483,8484,8485

# Launch 2 instances, then immediately start training
# (only use -StartTraining if the map auto-loads, e.g. with TMLoader profile)
.\scripts\launch_multienv.ps1 -StartTraining

# If TmForever.exe is not auto-detected:
.\scripts\launch_multienv.ps1 -TmExe "D:\Games\TmForever.exe"

# Increase wait time if TM loads slowly on your machine
.\scripts\launch_multienv.ps1 -WaitSeconds 20
```

### What the script does

```
For each port in [8483, 8484, ...]:
  1. Patch Python_Link.as  →  RegisterVariable("custom_port", PORT)
  2. Copy to %DOCUMENTS%\TMInterface\Plugins\Python_Link.as
  3. Launch TmForever.exe
  4. Wait N seconds  (TM loads, plugin binds port)

Restore original Python_Link.as
```

Because each TM instance reads the plugin **once at startup**, changing the file between launches is safe — instance 1 keeps its port in memory while instance 2 loads with the next port.

### After the script

1. In each TM window, load your training map.
2. Confirm in the TMInterface log that each window shows its assigned port.
3. Run training:
   ```powershell
   python main.py --ports 8483 8484
   ```

---

## Method 3 — Manual one-time setup per instance

Use this if the other methods don't work or if you want to control the process fully.

### How to set the port manually

The plugin reads a TMInterface variable called `custom_port` at startup. For the variable to be set **before** `Main()` runs, it must come from a TMLoader configstring or from TMInterface's autoexec (if your version supports one).

#### Via TMLoader configstring

If you use TMLoader to launch TM, add a configstring argument:

```
run TmForever "MyProfile" /configstring="set custom_port 8484"
```

For two instances:
- Instance 1: `run TmForever "MyProfile" /configstring="set custom_port 8483"`
- Instance 2: `run TmForever "MyProfile" /configstring="set custom_port 8484"`

#### Via manual plugin copies (no TMLoader, no scripts)

1. Make a second copy of the plugin:
   ```
   Python_Link_8483.as   ← original, port 8483
   Python_Link_8484.as   ← copy, edit RegisterVariable default to 8484
   ```

2. Rename the copy to `Python_Link.as` in the Plugins folder before launching each instance.

3. Launch TM instance 1 → rename to 8484 → launch instance 2 → restore original.

This is exactly what `launch_multienv.ps1` automates.

---

## Quick reference

| I want… | Use |
|---------|-----|
| Easiest possible setup | Method 1 (auto-scan, just open TM windows in order) |
| Fully automated launch | Method 2 (`launch_multienv.ps1`) |
| TMLoader workflow | Method 3 (configstring) |
| Verify ports are assigned | Open F3 (TMInterface log) in each TM window |

---

## Training commands

```powershell
# 2 environments
python main.py --ports 8483 8484

# 2 environments, debug mode (TensorBoard updates within ~1 minute)
python main.py --ports 8483 8484 --debug

# 4 environments
python main.py --ports 8483 8484 8485 8486 --timesteps 2000000

# TensorBoard (separate terminal)
tensorboard --logdir tensorboard
```

---

## How spawn strategy works

Each reset, the environment picks one of three spawn strategies:

| Strategy | Probability | Requires |
|----------|-------------|---------|
| **History rewind** | 40% | ≥100 saved states in memory |
| **TP + warm-up** | 40% | `data/map_blocks.json` loaded |
| **Start-line restart** | 20% | always available |

**History rewind** — `rewind_to_state()` teleports the car back to a real physics snapshot from earlier in training. The car's speed and orientation are exactly as they were.

**TP + warm-up** — uses `execute_command("tp X Y Z")` to jump to a random block along the centreline, then presses gas for 1–25 steps at 5× simulation speed to produce a varied starting speed (~5–75 km/h). The block chosen is curriculum-gated: early in training it spawns near the start; as `state_history` fills up it can spawn up to 60% along the track.

**Start-line restart** — `give_up()` returns the car to the start block.

---

## Performance tips

- **2 instances** is usually the sweet spot for a single gaming PC (one CPU core per env worker, plus the learner on the main thread).
- Use `--debug` while tuning to confirm TensorBoard is receiving data before committing to long runs.
- Each TM instance should be visible (not minimised) — some Openplanet builds throttle hidden windows.
- If one env consistently lags, reduce `ticks_per_step` in `TrackmaniaEnv.__init__` for that port.
