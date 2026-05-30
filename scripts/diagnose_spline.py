"""
Diagnose the spline path near the start line.

Shows:
 1. First 30 path points and their arc lengths
 2. Whether they form a sensible forward path
 3. What a test projection at the start looks like
 4. Whether the windowed search works correctly vs full search
"""
import json, math, os, sys, bisect
import numpy as np

# ---- Load blocks ----
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
blocks_path = os.path.join(data_dir, "map_blocks.json")

with open(blocks_path) as f:
    raw_blocks = json.load(f)

# ---- Replicate env.py's path building ----
def sort_blocks(blocks):
    return blocks

def build_path_points(blocks):
    return [(b["world_center"]["x"], b["world_center"]["y"], b["world_center"]["z"]) for b in blocks]

def compute_arc(points):
    arc = [0.0]
    for i in range(1, len(points)):
        d = math.sqrt(sum((a-b)**2 for a, b in zip(points[i], points[i-1])))
        arc.append(arc[-1] + d)
    return arc

sorted_blocks = sort_blocks(raw_blocks)
path_points = build_path_points(sorted_blocks)
path_arc = compute_arc(path_points)

print("=" * 70)
print("SPLINE DIAGNOSTIC")
print("=" * 70)
print(f"Total blocks: {len(sorted_blocks)}")
print(f"Total arc length: {path_arc[-1]:.0f} m")
print()

# ---- Show first 30 points ----
print("First 30 path points:")
print(f"{'Idx':>4} {'Name':>40} {'X':>8} {'Y':>6} {'Z':>8} {'Arc':>8} {'Gap':>6}")
for i in range(min(30, len(sorted_blocks))):
    b = sorted_blocks[i]
    c = b["world_center"]
    gap = path_arc[i] - path_arc[i-1] if i > 0 else 0.0
    print(f"{i:4d} {b['name'][:40]:>40} {c['x']:8.0f} {c['y']:6.0f} {c['z']:8.0f} {path_arc[i]:8.1f} {gap:6.1f}")

print()

# ---- Show last 10 points (where the bad blocks are) ----
print("Last 10 path points:")
for i in range(max(0, len(sorted_blocks)-10), len(sorted_blocks)):
    b = sorted_blocks[i]
    c = b["world_center"]
    gap = path_arc[i] - path_arc[i-1] if i > 0 else 0.0
    print(f"{i:4d} {b['name'][:40]:>40} {c['x']:8.0f} {c['y']:6.0f} {c['z']:8.0f} {path_arc[i]:8.1f} {gap:6.1f}")

print()

# ---- Test projection at the start ----
print("=" * 70)
print("PROJECTION TEST")
print("=" * 70)

# Precompute numpy arrays like env.py does
path_A = np.array(path_points[:-1], dtype=np.float64)
path_B = np.array(path_points[1:], dtype=np.float64)
path_AB = path_B - path_A
path_seg_lens_sq = np.maximum(np.sum(path_AB**2, axis=1), 1e-9)
path_arc_np = np.array(path_arc[:-1], dtype=np.float64)
path_seg_lens = np.sqrt(path_seg_lens_sq)

def project(pos, hint_arc=None, search_radius=500.0):
    n_segs = len(path_A)
    lo_seg, hi_seg = 0, n_segs
    if hint_arc is not None:
        lo_arc = max(0.0, hint_arc - search_radius)
        hi_arc = hint_arc + search_radius
        lo_seg = max(0, bisect.bisect_right(path_arc, lo_arc) - 1)
        hi_seg = min(n_segs, bisect.bisect_left(path_arc, hi_arc))
        if hi_seg <= lo_seg:
            lo_seg, hi_seg = 0, n_segs
    P = np.array(pos, dtype=np.float64)
    A_win = path_A[lo_seg:hi_seg]
    AB_win = path_AB[lo_seg:hi_seg]
    lens_sq_win = path_seg_lens_sq[lo_seg:hi_seg]
    AP = P - A_win
    t = np.sum(AP * AB_win, axis=1) / lens_sq_win
    t = np.clip(t, 0.0, 1.0)
    C = A_win + t[:, np.newaxis] * AB_win
    dists_sq = np.sum((P - C)**2, axis=1)
    best_idx = int(np.argmin(dists_sq))
    global_idx = lo_seg + best_idx
    min_dist = float(np.sqrt(dists_sq[best_idx]))
    best_arc = float(path_arc_np[global_idx] + t[best_idx] * path_seg_lens[global_idx])
    return global_idx, min_dist, best_arc

# Start line position (from block 0)
start = sorted_blocks[0]["world_center"]
start_pos = (start["x"], start["y"], start["z"])

print(f"\nStart line position: {start_pos}")

# Full search from start
seg, dist, arc = project(start_pos)
print(f"Full search:     seg={seg}, dist={dist:.2f}m, arc={arc:.1f}m")

# Windowed from 0
seg, dist, arc = project(start_pos, hint_arc=0.0, search_radius=500.0)
print(f"Window (hint=0): seg={seg}, dist={dist:.2f}m, arc={arc:.1f}m")

# Simulate driving 50m forward - approximate by using block 2's center
if len(sorted_blocks) > 2:
    b2 = sorted_blocks[2]["world_center"]
    b2_pos = (b2["x"], b2["y"], b2["z"])
    print(f"\nBlock 2 position: {b2_pos}")
    
    seg_f, dist_f, arc_f = project(b2_pos)
    print(f"Full search:     seg={seg_f}, dist={dist_f:.2f}m, arc={arc_f:.1f}m")
    
    seg_w, dist_w, arc_w = project(b2_pos, hint_arc=0.0, search_radius=500.0)
    print(f"Window (hint=0): seg={seg_w}, dist={dist_w:.2f}m, arc={arc_w:.1f}m")

# Test blocks 3-10 with progressive windowing (simulating what happens during training)
print(f"\n{'='*70}")
print("SIMULATED PROGRESSIVE DRIVE (blocks 0-15)")
print(f"{'='*70}")
print(f"{'Block':>5} {'Name':>35} {'FullArc':>8} {'WinArc':>8} {'FullDist':>8} {'WinDist':>8} {'WinSegs':>8}")

current_arc = 0.0
for i in range(min(16, len(sorted_blocks))):
    b = sorted_blocks[i]["world_center"]
    pos = (b["x"], b["y"], b["z"])
    
    _, dist_full, arc_full = project(pos)
    
    # Windowed using current progress cursor
    lo_arc = max(0.0, current_arc - 500.0)
    hi_arc = current_arc + 500.0
    lo_seg = max(0, bisect.bisect_right(path_arc, lo_arc) - 1)
    hi_seg = min(len(path_A), bisect.bisect_left(path_arc, hi_arc))
    n_segs_window = hi_seg - lo_seg
    
    _, dist_win, arc_win = project(pos, hint_arc=current_arc, search_radius=500.0)
    
    # Update cursor like the reward function would
    if arc_win > current_arc and (arc_win - current_arc) <= 50.0:
        current_arc = arc_win
    
    name = sorted_blocks[i]["name"][:35]
    print(f"{i:5d} {name:>35} {arc_full:8.1f} {arc_win:8.1f} {dist_full:8.2f} {dist_win:8.2f} {n_segs_window:8d}")

print(f"\nFinal progress cursor: {current_arc:.1f} m")
print(f"Expected: ~{path_arc[min(15, len(path_arc)-1)]:.1f} m")
