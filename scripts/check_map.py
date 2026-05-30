"""Quick diagnostic: check if track start and finish are physically close."""
import json, math, os

blocks = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "map_blocks.json")))
s = blocks[0]["world_center"]
e = blocks[-1]["world_center"]
d = math.sqrt((s["x"]-e["x"])**2 + (s["z"]-e["z"])**2)

print(f"Total blocks: {len(blocks)}")
print(f"Start block: ({s['x']:.0f}, {s['y']:.0f}, {s['z']:.0f})  name={blocks[0]['name']}")
print(f"End   block: ({e['x']:.0f}, {e['y']:.0f}, {e['z']:.0f})  name={blocks[-1]['name']}")
print(f"XZ distance start<->end: {d:.0f} m")
print()

# Check for large gaps (blocks that jump far from predecessor)
print("Segments with large gaps (>70m):")
for i in range(1, len(blocks)):
    p = blocks[i-1]["world_center"]
    c = blocks[i]["world_center"]
    gap = math.sqrt((c["x"]-p["x"])**2 + (c["z"]-p["z"])**2)
    if gap > 70:
        print(f"  Block {i-1} -> {i}: {gap:.0f} m gap")
        print(f"    {blocks[i-1]['name'][:50]} -> {blocks[i]['name'][:50]}")
