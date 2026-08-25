"""
sensor_simulator.py
--------------------
Simulates real-time sensor data for a textile machine.

Instead of pure random noise, each reading is generated with:
  - Mean-reverting random walk (so values drift realistically instead of
    jumping around) around a baseline "normal operating" value.
  - Occasional "anomaly" episodes (machine stress) that push temperature,
    vibration, and energy usage up and raise defect probability — mimicking
    a real machine that occasionally runs hot / vibrates more / makes bad
    product for a few cycles before settling back down.
  - Cumulative production_count and defect_count, like a real counter would
    behave (only ever goes up).
"""

import random
from datetime import datetime

import numpy as np


class TextileMachineSimulator:
    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Baseline ("normal operating") values — tune these to match your machine
        self.baseline = {
            "speed": 1550.0,
            "temperature": 78.0,
            "humidity": 45.0,
            "vibration": 0.45,
            "energy": 105.0,
        }

        # Current state, starts at baseline
        self.speed = self.baseline["speed"]
        self.temperature = self.baseline["temperature"]
        self.humidity = self.baseline["humidity"]
        self.vibration = self.baseline["vibration"]
        self.energy = self.baseline["energy"]

        self.production_count = 0
        self.defect_count = 0

        self.anomaly_active = False
        self.anomaly_ticks_left = 0

    def _maybe_trigger_anomaly(self):
        """Randomly start/stop a short 'machine stress' episode."""
        if not self.anomaly_active and random.random() < 0.02:  # ~2% chance per tick
            self.anomaly_active = True
            self.anomaly_ticks_left = random.randint(3, 8)

        if self.anomaly_active:
            self.anomaly_ticks_left -= 1
            if self.anomaly_ticks_left <= 0:
                self.anomaly_active = False

    def next_reading(self) -> dict:
        """Generate the next simulated sensor reading."""
        self._maybe_trigger_anomaly()

        anomaly_boost = 1.5 if self.anomaly_active else 1.0
        defect_prob = 0.22 if self.anomaly_active else 0.03

        # Mean-reverting random walk: noise + gentle pull back toward baseline
        self.speed += np.random.normal(0, 8) + (self.baseline["speed"] - self.speed) * 0.02
        self.speed = float(np.clip(self.speed, 1300, 1800))

        self.temperature += (
            np.random.normal(0, 0.6) * anomaly_boost
            + (self.baseline["temperature"] - self.temperature) * 0.03
        )
        self.temperature = float(np.clip(self.temperature, 60, 95))

        self.humidity += np.random.normal(0, 1.2) + (self.baseline["humidity"] - self.humidity) * 0.03
        self.humidity = float(np.clip(self.humidity, 20, 70))

        self.vibration += (
            np.random.normal(0, 0.02) * anomaly_boost
            + (self.baseline["vibration"] - self.vibration) * 0.05
        )
        self.vibration = float(np.clip(self.vibration, 0.1, 1.2))

        self.energy += (
            np.random.normal(0, 3) * anomaly_boost
            + (self.baseline["energy"] - self.energy) * 0.02
        )
        self.energy = float(np.clip(self.energy, 70, 160))

        produced = random.randint(1, 4)
        self.production_count += produced
        if random.random() < defect_prob:
            self.defect_count += 1

        return {
            "timestamp": datetime.now(),
            "machine_speed_rpm": round(self.speed, 2),
            "temperature_c": round(self.temperature, 2),
            "humidity_percent": round(self.humidity, 2),
            "vibration_level": round(self.vibration, 4),
            "energy_usage_kwh": round(self.energy, 2),
            "production_count": self.production_count,
            "defect_count": self.defect_count,
        }


if __name__ == "__main__":
    # Quick standalone test: print 5 readings
    sim = TextileMachineSimulator(seed=42)
    for _ in range(5):
        print(sim.next_reading())