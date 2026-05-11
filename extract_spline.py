"""
Extract road blocks from a Trackmania GBX map file and write them to
data/map_blocks.json in path order (start → road → finish).

Usage:
    python extract_spline.py                        # uses data/map.gbx
    python extract_spline.py path/to/MyMap.gbx      # custom map

The JSON is consumed by interfacing/game_env.py to build the centerline
used for the arc-length progress reward.
"""

import json
import math
import os
import sys

import pygbx


# ---------------------------------------------------------------------------
# Block classification helpers
# ---------------------------------------------------------------------------

VALID_KEYWORDS = ["road", "start", "finish", "checkpoint", "circuit", "pipe", "platform"]

def _is_valid_block(name: str) -> bool:
    """Return True for road/structural blocks; skip decorative scenery."""
    n = name.lower()
    return any(kw in n for kw in VALID_KEYWORDS)

def _is_start(name: str) -> bool:
    return "start" in name.lower()

def _is_finish(name: str) -> bool:
    return "finish" in name.lower() and "start" not in name.lower()

def _is_checkpoint(name: str) -> bool:
    return "checkpoint" in name.lower()


# ---------------------------------------------------------------------------
# Path ordering
# ---------------------------------------------------------------------------

def _sort_blocks_greedy(blocks):
    """
    Order blocks so the path runs start → road/checkpoints → finish.

    1. Start block(s) are placed first (the one closest to grid origin wins
       as a tiebreaker).
    2. Finish block(s) are pinned to the end.
    3. Everything in between is ordered with greedy nearest-neighbour in the
       X-Z plane (ignores vertical variation for flatter track representations).

    This avoids the common failure mode where the greedy chain doubles back
    near the start line.
    """
    starts   = [b for b in blocks if _is_start(b["name"])]
    finishes = [b for b in blocks if _is_finish(b["name"])]
    middles  = [b for b in blocks if not _is_start(b["name"]) and not _is_finish(b["name"])]

    if not starts:
        # Fallback: use the block closest to the world origin as start.
        blocks_copy = list(blocks)
        blocks_copy.sort(key=lambda b: b["world_center"]["x"]**2 + b["world_center"]["z"]**2)
        starts = [blocks_copy[0]]
        middles = [b for b in blocks if b is not starts[0] and not _is_finish(b["name"])]

    # If there are multiple start blocks, take the one closest to origin.
    starts.sort(key=lambda b: b["world_center"]["x"]**2 + b["world_center"]["z"]**2)
    seed = starts[0]
    remaining = middles + starts[1:]  # extra start blocks treated as road

    ordered = [seed]
    while remaining:
        last = ordered[-1]["world_center"]
        best_i, best_d = 0, float("inf")
        for i, b in enumerate(remaining):
            c = b["world_center"]
            d = math.sqrt((last["x"] - c["x"])**2 + (last["z"] - c["z"])**2)
            if d < best_d:
                best_d, best_i = d, i
        ordered.append(remaining.pop(best_i))

    # Append finish block(s) at the very end.
    ordered.extend(finishes)
    return ordered


def _compute_arc_lengths(ordered_blocks):
    """Return cumulative 3-D distances along the sorted block list."""
    arc = [0.0]
    for i in range(1, len(ordered_blocks)):
        prev = ordered_blocks[i - 1]["world_center"]
        cur  = ordered_blocks[i]["world_center"]
        d = math.sqrt(
            (cur["x"] - prev["x"])**2 +
            (cur["y"] - prev["y"])**2 +
            (cur["z"] - prev["z"])**2
        )
        arc.append(arc[-1] + d)
    return arc


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_map_blocks(gbx_path: str, output_path: str):
    print(f"Reading map: {gbx_path}")

    try:
        g = pygbx.Gbx(gbx_path)
    except Exception as e:
        print(f"Failed to read map: {e}")
        return

    challenges = g.get_classes_by_ids([pygbx.GbxType.CHALLENGE])
    if not challenges:
        print("No Challenge class found in GBX file!")
        return

    challenge = challenges[0]
    blocks = challenge.blocks
    print(f"Found {len(blocks)} raw blocks in the map.")

    extracted = []
    for b in blocks:
        if not _is_valid_block(b.name):
            continue

        pos = b.position

        # TM block grid: 32 × 32 m footprint, 8 m vertical unit.
        # The centre of a standard 1-unit-tall block is at the midpoint of
        # its vertical extent, i.e.  Y_base + 4.0 m.
        real_x = pos.x * 32.0 + 16.0
        real_y = pos.y * 8.0 + 4.0   # +4 = vertical centre of a 1-unit block
        real_z = pos.z * 32.0 + 16.0

        extracted.append({
            "name": b.name,
            "grid": {"x": int(pos.x), "y": int(pos.y), "z": int(pos.z)},
            "world_center": {"x": real_x, "y": real_y, "z": real_z},
            "rotation": int(b.rotation) if hasattr(b, "rotation") else 0,
            "is_start":      _is_start(b.name),
            "is_finish":     _is_finish(b.name),
            "is_checkpoint": _is_checkpoint(b.name),
        })

    if not extracted:
        print("No valid road/structural blocks found.  Check VALID_KEYWORDS.")
        return

    print(f"Extracted {len(extracted)} valid blocks.")

    # Sort into path order and annotate with arc-length.
    ordered = _sort_blocks_greedy(extracted)
    arc_lengths = _compute_arc_lengths(ordered)
    for block, arc in zip(ordered, arc_lengths):
        block["arc_length"] = round(arc, 3)

    total_m = arc_lengths[-1] if arc_lengths else 0.0
    print(f"Path ordered: {len(ordered)} blocks, total arc length ≈ {total_m:.0f} m")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ordered, f, indent=2)

    print(f"Written to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        gbx_file = sys.argv[1]
    else:
        gbx_file = os.path.join(repo_root, "data", "map.gbx")

    out_file = os.path.join(repo_root, "data", "map_blocks.json")
    extract_map_blocks(gbx_file, out_file)
