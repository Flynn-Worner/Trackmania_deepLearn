import pygbx
import json
import os
import sys

def extract_map_blocks(gbx_path, output_path):
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
    
    print(f"Found {len(blocks)} blocks in the map.")
    
    extracted_blocks = []
    
    for b in blocks:
        # Ignore purely decorative blocks or grass (keep road/tech)
        # Often block names are like 'RoadTechStraight', 'RoadDirtCurve3', 'ArenaStart'
        name = b.name.lower()
        if 'road' not in name and 'start' not in name and 'finish' not in name and 'checkpoint' not in name:
            # We skip decorative scenery for the centerline calculation
            continue
            
        # Trackmania block grid size is 32x32m, height is 8m
        # Grid origin is (0,0,0). A block at X=5, Y=2, Z=10
        # The center of the block physically is (X*32 + 16, Y*8, Z*32 + 16)
        # Note: Y is height in TM, but pygbx `b.pos` usually maps x, y, z.
        
        pos = b.position
        real_x = pos.x * 32.0 + 16.0
        real_y = pos.y * 8.0          # Base height of the block
        real_z = pos.z * 32.0 + 16.0
        
        extracted_blocks.append({
            "name": b.name,
            "grid": {"x": pos.x, "y": pos.y, "z": pos.z},
            "world_center": {"x": real_x, "y": real_y, "z": real_z},
            "rotation": b.rotation
        })

    with open(output_path, "w") as f:
        json.dump(extracted_blocks, f, indent=4)
        
    print(f"Successfully extracted {len(extracted_blocks)} valid road blocks to {output_path}")

if __name__ == "__main__":
    gbx_file = os.path.join(os.path.dirname(__file__), "data", "map.gbx")
    out_file = os.path.join(os.path.dirname(__file__), "data", "map_blocks.json")
    extract_map_blocks(gbx_file, out_file)
