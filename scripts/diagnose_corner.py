"""Analyze the track geometry near the 210m crash zone."""
import json, math, os

blocks = json.load(open('data/map_blocks.json'))

def sort_blocks(blocks):
    start = next((b for b in blocks if 'Start' in b['name']), blocks[0])
    sorted_b = [start]
    remaining = [b for b in blocks if b is not start]
    while remaining:
        last = sorted_b[-1]['world_center']
        best_i, best_d = 0, float('inf')
        for i, b in enumerate(remaining):
            c = b['world_center']
            d = math.sqrt((c['x']-last['x'])**2 + (c['y']-last['y'])**2 + (c['z']-last['z'])**2)
            if d < best_d:
                best_d, best_i = d, i
        sorted_b.append(remaining.pop(best_i))
    return sorted_b

def compute_arc(blocks):
    arc = [0.0]
    for i in range(1, len(blocks)):
        p, c = blocks[i-1]['world_center'], blocks[i]['world_center']
        d = math.sqrt((c['x']-p['x'])**2 + (c['y']-p['y'])**2 + (c['z']-p['z'])**2)
        arc.append(arc[-1] + d)
    return arc

sb = sort_blocks(blocks)
arc = compute_arc(sb)

print("Blocks near the 210m crash zone (blocks 10-22):")
print(f"{'Idx':>4} {'Arc':>7} {'X':>6} {'Z':>6} {'Name':>30}  Turn")
for i in range(10, min(22, len(sb))):
    c = sb[i]['world_center']
    turn_str = ""
    if 0 < i < len(sb) - 1:
        p = sb[i-1]['world_center']
        n = sb[i+1]['world_center']
        v1x, v1z = c['x'] - p['x'], c['z'] - p['z']
        v2x, v2z = n['x'] - c['x'], n['z'] - c['z']
        l1 = math.sqrt(v1x**2 + v1z**2)
        l2 = math.sqrt(v2x**2 + v2z**2)
        if l1 > 0 and l2 > 0:
            dot = (v1x * v2x + v1z * v2z) / (l1 * l2)
            angle = math.degrees(math.acos(max(-1, min(1, dot))))
            turn_str = f"  {angle:.0f} deg"
    print(f"{i:4d} {arc[i]:7.1f} {c['x']:6.0f} {c['z']:6.0f} {sb[i]['name'][:30]:>30}{turn_str}")

print()
print("=" * 60)
print("REWARD ANALYSIS: why speed-crashing beats surviving")
print("=" * 60)
print()
print("Typical episode that crashes at 210m:")
print("  speed reward:    +27  (avg 110 km/h over 200 steps)")
print("  progress reward: +21  (210m * 0.10)")
print("  crash penalty:   -5")
print("  NET:             +43")
print()
print("Hypothetical episode that slows to 60 km/h and survives to 300m:")
print("  speed reward:    +25  (avg 60 km/h over 250 steps)")
print("  progress reward: +30  (300m * 0.10)")
print("  crash penalty:    0")
print("  NET:             +55  (only slightly better!)")
print()
print("The speed reward is SO dominant that the agent is rewarded")
print("almost equally for crashing fast as for surviving slow.")
print("Fix: make progress the clear #1 signal, reduce speed to minor.")
