"""TCP bridge for telemetry.as plugin I/O."""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket


@dataclass
class TelemetryState:
    position: tuple[float, float, float]
    speed_mps: float
    yaw: float
    sample_idx: int

    @property
    def display_speed(self) -> float:
        # rewards/obs code expects km/h-style display speed
        return self.speed_mps * 3.6

    @property
    def yaw_pitch_roll(self) -> tuple[float, float, float]:
        return (self.yaw, 0.0, 0.0)

    @property
    def velocity(self) -> tuple[float, float, float]:
        fx = -math.sin(self.yaw)
        fz = math.cos(self.yaw)
        return (fx * self.speed_mps, 0.0, fz * self.speed_mps)


class TelemetryBridge:
    """Host-side socket server for the Openplanet telemetry.as client."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, accept_timeout: float = 180.0):
        self.host = host
        self.port = port
        self.accept_timeout = accept_timeout
        self.server: socket.socket | None = None
        self.conn: socket.socket | None = None
        self._buffer = b""
        self._sample_idx = 0

    def connect(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.server.settimeout(self.accept_timeout)

        conn, _addr = self.server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(5.0)
        self.conn = conn

    def send_action(self, action: str) -> None:
        if self.conn is None:
            raise RuntimeError("Bridge is not connected")
        payload = action.strip().encode("ascii", errors="ignore")[:16]
        self.conn.sendall(payload)

    def _readline(self) -> str:
        if self.conn is None:
            raise RuntimeError("Bridge is not connected")

        while b"\n" not in self._buffer:
            chunk = self.conn.recv(1024)
            if not chunk:
                raise ConnectionError("telemetry.as disconnected")
            self._buffer += chunk

        raw, self._buffer = self._buffer.split(b"\n", 1)
        return raw.decode("ascii", errors="ignore").strip()

    def recv_state(self) -> TelemetryState:
        while True:
            line = self._readline()
            parts = line.split(",")
            if len(parts) != 5:
                continue
            try:
                x, y, z, speed_mps, yaw = [float(v) for v in parts]
            except ValueError:
                continue

            self._sample_idx += 1
            return TelemetryState(
                position=(x, y, z),
                speed_mps=speed_mps,
                yaw=yaw,
                sample_idx=self._sample_idx,
            )

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                pass
            self.server = None