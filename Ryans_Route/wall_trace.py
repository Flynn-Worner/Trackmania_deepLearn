import socket
import time
import json
import os
import keyboard
import numpy as np


HOST = "127.0.0.1"
PORT = 9000
RECORD_INTERVAL = 0.3
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_telemetry(message):
    x, y, z, speed, yaw = map(float, message.split(","))
    return x, y, z, speed, yaw


def save_wall_json(points_array, label):
    os.makedirs(DATA_DIR, exist_ok=True)
    json_path = os.path.join(DATA_DIR, f"{label}_wall_points.json")

    payload = {
        "version": 1,
        "wall": label,
        "record_interval_s": RECORD_INTERVAL,
        "num_points": int(len(points_array)),
        "points_xz": points_array.tolist(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved JSON wall points to {json_path}")


def save_combined_json_if_available():
    left_path = os.path.join(DATA_DIR, "left_wall_points.json")
    right_path = os.path.join(DATA_DIR, "right_wall_points.json")

    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        return

    with open(left_path, "r", encoding="utf-8") as f:
        left = json.load(f)
    with open(right_path, "r", encoding="utf-8") as f:
        right = json.load(f)

    combined = {
        "version": 1,
        "left_wall_points_xz": left.get("points_xz", []),
        "right_wall_points_xz": right.get("points_xz", []),
        "left_count": len(left.get("points_xz", [])),
        "right_count": len(right.get("points_xz", [])),
    }

    combined_path = os.path.join(DATA_DIR, "wall_points.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined wall JSON to {combined_path}")


def record_trace(label):
    points = []
    recording = False
    last_record_time = 0

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Waiting for TMInterface plugin on {HOST}:{PORT}...")
    conn, addr = server.accept()
    print(f"Connected by {addr}")

    print(f"Recording {label} wall trace.")
    print("Press R to start.")
    print("Drive along the wall.")
    print("Press T to stop and save.")

    try:
        while True:
            data = conn.recv(1024)

            if not data:
                print("Connection closed.")
                break

            message = data.decode("utf-8").strip().split("\n")[-1]

            try:
                x, y, z, speed, yaw = parse_telemetry(message)
            except ValueError:
                conn.sendall("MANUAL".encode("utf-8"))
                continue

            if keyboard.is_pressed("r") and not recording:
                recording = True
                last_record_time = 0
                print("Recording started.")

            if keyboard.is_pressed("t") and recording:
                print("Recording stopped.")
                break

            if recording:
                now = time.time()

                if now - last_record_time >= RECORD_INTERVAL:
                    points.append([x, z])
                    last_record_time = now
                    print(f"{label} point {len(points)}: x={x:.2f}, z={z:.2f}")

            conn.sendall("MANUAL".encode("utf-8"))

    finally:
        conn.close()
        server.close()

    points = np.array(points, dtype=np.float32)

    if len(points) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        npy_path = os.path.join(DATA_DIR, f"{label}_wall_points.npy")
        np.save(npy_path, points)
        print(f"Saved {len(points)} points to {npy_path}")
        save_wall_json(points, label)
        save_combined_json_if_available()
    else:
        print("No points recorded.")


if __name__ == "__main__":
    side = input("Which wall are you recording? Type left or right: ").strip().lower()

    if side == "left":
        record_trace("left")
    elif side == "right":
        record_trace("right")
    else:
        print("Invalid option. Type left or right.")