"""Check what's at the 316.8m crash zone."""
import json, math

blocks = json.load(open('data/map_blocks.json'))

def sort_blocks(blocks):
    start = next((b for b in blocks if 'Start' in b['name']), blocks[0])
    sb = [start]
    rem = [b for b in blocks if b is not start]
    while rem:
        last = sb[-1]['world_center']
        best_i, best_d = 0, float('inf')
        for i, b in enumerate(rem):
            c = b['world_center']
            d = math.sqrt((c['x']-last['x'])**2+(c['y']-last['y'])**2+(c['z']-last['z'])**2)
            if d < best_d:
                best_d, best_i = d, i
        sb.append(rem.pop(best_i))
    return sb

def compute_arc(blocks):
    a = [0.0]
    for i in range(1, len(blocks)):
        p = blocks[i-1]['world_center']
        c = blocks[i]['world_center']
        a.append(a[-1] + math.sqrt((c['x']-p['x'])**2+(c['y']-p['y'])**2+(c['z']-p['z'])**2))
    return a

sb = sort_blocks(blocks)
a = compute_arc(sb)

print("Blocks 18-30 (around the 316m crash zone):")
for i in range(18, min(30, len(sb))):
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
            ang = math.degrees(math.acos(max(-1, min(1, dot))))
            turn_str = f"  turn={ang:.0f} deg"
    pos = f"({c['x']:.0f}, {c['z']:.0f})"
    print(f"  {i:3d}  arc={a[i]:6.1f}m  {pos:>14}  {sb[i]['name'][:25]}{turn_str}")
