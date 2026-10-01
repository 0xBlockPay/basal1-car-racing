"""
Basal-1 Autonomous Driving Agent for OpenAI Gym Box2D CarRacing (v2 / v3)
Endpoint: POST /v1/systemone
Features:
  - Corrected road visual perception (scans ahead of the hood, avoids red car body)
  - Active lane centering with Basal decision fusion
  - Corner speed governor (82 km/h straights, 52 km/h technical turns)
  - Active grass recovery without stall
"""

import gymnasium as gym
import numpy as np
import requests
import math
import time

# ==============================================================================
# 1. Basal-1 System One Client (POST /v1/systemone)
# ==============================================================================
class Basal:
    """
    Client for communicating with the Basal-1 System One REST endpoint.
    Handles connection probing, prompt dispatching, and score parsing.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/systemone"
        self.is_connected, self.connection_msg = self._test_connection()
        self.last_latency_ms = 0.0
        self.last_source = "API" if self.is_connected else "LOCAL-SIM"

    def _test_connection(self) -> tuple[bool, str]:
        """Probes the target endpoint with a lightweight status question."""
        probe = {
            "state": "CarRacing connection probe",
            "questions": {
                "probe": {
                    "type": "choice",
                    "instructions": "Is system operational?",
                    "criteria": {"ok": "Operational", "wait": "Wait"}
                }
            }
        }
        candidates = [
            self.endpoint,
            "http://localhost:8000/v1/systemone" if "127.0.0.1" in self.endpoint else "http://127.0.0.1:8000/v1/systemone"
        ]
        for url in candidates:
            try:
                res = requests.post(url, json=probe, timeout=2.5)
                if res.status_code == 200:
                    self.endpoint = url
                    return True, f"Connected to {url}"
            except Exception:
                pass
        return False, "Could not connect to port 8000"

    def evaluate_car_controls(self, context: str) -> dict:
        """
        Sends the situation context to /v1/systemone and parses steering and throttle.
        """
        start_time = time.perf_counter()

        steer_weights = {
            "hard_left": -1.0,
            "gentle_left": -0.6,
            "straight": 0.0,
            "gentle_right": 0.6,
            "hard_right": 1.0
        }
        throttle_weights = {
            "zero": 0.0,
            "light": 0.35,
            "moderate": 0.60,
            "strong": 0.85,
            "full": 1.00
        }

        if self.is_connected:
            try:
                payload = {
                    "state": context,
                    "questions": {
                        "steer": {
                            "type": "choice",
                            "instructions": "Which steering direction should the car wheels take to follow the road?",
                            "criteria": {
                                "hard_left": "Sharp turn left (corner or drifting right)",
                                "gentle_left": "Gentle turn left back to center",
                                "straight": "Centered straight",
                                "gentle_right": "Gentle turn right back to center",
                                "hard_right": "Sharp turn right (corner or drifting left)"
                            }
                        },
                        "throttle": {
                            "type": "choice",
                            "instructions": "How much throttle acceleration should be applied?",
                            "criteria": {
                                "zero": "Coast with no gas",
                                "light": "Light throttle 35% for cornering",
                                "moderate": "Moderate throttle 60%",
                                "strong": "Strong acceleration 85%",
                                "full": "Full flat out throttle 100%"
                            }
                        }
                    }
                }
                res = requests.post(self.endpoint, json=payload, timeout=2.0)
                if res.status_code == 200:
                    data = res.json()
                    self.last_latency_ms = (time.perf_counter() - start_time) * 1000.0
                    self.last_source = "API"

                    def extract_res(root, q_name):
                        if not isinstance(root, dict): return None
                        if q_name in root: return root[q_name]
                        for n in ["answers", "questions", "results", "choices"]:
                            if n in root and isinstance(root[n], dict) and q_name in root[n]:
                                return root[n][q_name]
                        return None

                    def parse_val(q_data, weights, default=0.0):
                        if q_data is None: return default
                        if isinstance(q_data, str):
                            for k, w in weights.items():
                                if k in q_data.lower(): return w
                            return default
                        if isinstance(q_data, dict):
                            for pk in ["probabilities", "probs", "scores"]:
                                if pk in q_data and isinstance(q_data[pk], dict):
                                    return sum(weights.get(k, 0.0) * float(p) for k, p in q_data[pk].items())
                            for ck in ["choice", "answer", "selected"]:
                                if ck in q_data and str(q_data[ck]).lower() in weights:
                                    return weights[str(q_data[ck]).lower()]
                        return default

                    steer = parse_val(extract_res(data, "steer"), steer_weights, default=0.0)
                    throttle = parse_val(extract_res(data, "throttle"), throttle_weights, default=0.5)
                    return {"steer": steer, "throttle": throttle}
                else:
                    self.is_connected = False
            except Exception:
                self.is_connected = False

        self.last_source = "LOCAL-SIM"
        self.last_latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {"steer": 0.0, "throttle": 0.5}

    def get_status_badge(self) -> str:
        if self.last_source == "API":
            return f"[API ONLINE | {self.last_latency_ms:4.1f}ms]"
        return "[LOCAL-SIM (Offline)]"


# Initialize Basal client instance
b = Basal("http://127.0.0.1:8000")


# ==============================================================================
# 2. Vision: Accurate Road Perception in Front of the Car (y=56, y=40)
# ==============================================================================
def is_road(pixel: np.ndarray) -> bool:
    """Road asphalt is gray: RGB values are close, brightness between 85 and 130."""
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    return abs(r - g) < 18 and abs(g - b) < 18 and 85 < r < 130


def extract_visual_telemetry(obs: np.ndarray, env: gym.Env):
    """
    Extracts road center, curvature, and vehicle offset by scanning pixels
    immediately in front of the car hood (y=56) and lookahead horizon (y=40).
    """
    speed = 45.0
    try:
        vx, vy = env.unwrapped.car.hull.linearVelocity
        speed = math.hypot(vx, vy) * 3.6
    except Exception:
        pass

    # Scan road width directly ahead of the car hood (y=56, avoids red car body at y=66)
    y_front = 56
    x_car = 48

    road_pixels_front = [x for x in range(5, 91) if is_road(obs[y_front, x])]

    if len(road_pixels_front) > 6:
        # Car is on track
        on_grass = False
        road_center_front = (road_pixels_front[0] + road_pixels_front[-1]) / 2.0
        offset = x_car - road_center_front  # >0: car is drifted to the right of centerline
    else:
        # Car is off-track on grass: scan full screen to locate asphalt
        on_grass = True
        all_road = [x for x in range(5, 91) if is_road(obs[y_front, x]) or is_road(obs[45, x])]
        if len(all_road) > 0:
            offset = x_car - float(np.mean(all_road))
        else:
            offset = 15.0

    # Lookahead horizon scan (y=40, ~20m ahead) to detect track curvature
    road_pixels_ahead = [x for x in range(5, 91) if is_road(obs[40, x])]
    if len(road_pixels_ahead) > 6:
        road_center_ahead = (road_pixels_ahead[0] + road_pixels_ahead[-1]) / 2.0
        curve = road_center_ahead - x_car  # <0 left curve, >0 right curve
    else:
        curve = 0.0

    direction_desc = (
        "sharp technical curve to the LEFT" if curve < -6 else
        "gentle curve to the LEFT" if curve < -2 else
        "sharp technical curve to the RIGHT" if curve > 6 else
        "gentle curve to the RIGHT" if curve > 2 else
        "straight open road"
    )
    position_desc = (
        f"drifting {abs(offset):.1f}m to the RIGHT of centerline" if offset > 1.5 else
        f"drifting {abs(offset):.1f}m to the LEFT of centerline" if offset < -1.5 else
        "centered on racing line"
    )

    context = (
        f"CarRacing Telemetry: Speed {speed:.1f} km/h. "
        f"Car is {'ON ASPHALT' if not on_grass else 'OFF-TRACK ON GRASS'}, {position_desc}. "
        f"Upcoming track is {direction_desc}. "
        f"Steer to maintain optimal racing line."
    )
    return context, speed, offset, curve, on_grass


# ==============================================================================
# 3. Simulation Loop with Action Repeat & Corner Speed Governor
# ==============================================================================
try:
    env = gym.make("CarRacing-v3", render_mode="human")
except Exception:
    env = gym.make("CarRacing-v2", render_mode="human")

obs, info = env.reset()
total_reward = 0.0

print("=" * 75)
print("  🏎️  BASAL-1 AUTOPILOT (Precision Steering & Speed Governor)")
print("=" * 75)
print(f"  Connection Status : {'[● ONLINE]' if b.is_connected else '[○ OFFLINE]'}")
print(f"  Target Endpoint   : {b.endpoint}")
print("=" * 75)

FRAME_SKIP = 3
current_action = np.array([0.0, 0.5, 0.0], dtype=np.float32)

for step in range(1500):
    context, speed, offset, curve, on_grass = extract_visual_telemetry(obs, env)

    if step % FRAME_SKIP == 0:
        controls = b.evaluate_car_controls(context)
        raw_steer = controls["steer"]
        raw_throttle = controls["throttle"]

        if on_grass:
            # Grass recovery: steer aggressively towards road with moderate throttle
            steer_action = -0.85 if offset > 0 else 0.85
            gas_action = 0.50
            brake_action = 0.0
        else:
            # On-track dynamics: blend Basal's decision with lane geometry
            geo_steer = -offset * 0.065 + curve * 0.075

            if abs(raw_steer) > 0.05:
                combined_steer = raw_steer * 0.6 + geo_steer * 0.7
            else:
                combined_steer = geo_steer * 1.1

            steer_action = float(np.clip(combined_steer, -1.0, 1.0))

            # Corner speed governor
            is_curve = abs(curve) > 3.5 or abs(offset) > 3.0
            max_allowed_speed = 52.0 if is_curve else 82.0

            if speed < 20.0:
                # Launch from standstill
                gas_action = 0.70
                brake_action = 0.0
            elif speed > max_allowed_speed + 8.0:
                # Approaching corner too fast: cut gas and apply brakes
                gas_action = 0.0
                brake_action = 0.45
            elif speed > max_allowed_speed:
                # Maintain governed speed
                gas_action = 0.15
                brake_action = 0.0
            else:
                # Accelerate towards top straight speed
                gas_action = max(raw_throttle, 0.45)
                brake_action = 0.0

        current_action = np.array([steer_action, gas_action, brake_action], dtype=np.float32)

    # Step physics in Gym Box2D
    obs, reward, terminated, truncated, info = env.step(current_action)
    total_reward += reward

    # Pygame HUD overlay
    try:
        import pygame
        if pygame.display.get_init():
            surf = pygame.display.get_surface()
            if surf:
                hud_font = pygame.font.SysFont("monospace", 12, bold=True)
                hud_msg = f"BASAL: {b.last_source} | STEER: {current_action[0]:+.2f} | GAS: {current_action[1]:.2f} | SPEED: {speed:.0f}km/h"
                hud_color = (60, 255, 130) if not on_grass else (255, 100, 100)
                txt = hud_font.render(hud_msg, True, hud_color)
                surf.blit(txt, (10, 8))
                pygame.display.flip()
    except Exception:
        pass

    if step % 15 == 0:
        status_text = "ON TRACK" if not on_grass else "GRASS RECOVERY"
        print(f"Step #{step:04d} | {status_text:<14} | Speed: {speed:4.1f} km/h | Steer: {current_action[0]:+5.2f} | Gas: {current_action[1]:4.2f} | Brake: {current_action[2]:4.2f} | Reward: {total_reward:6.1f}")

    if terminated or truncated:
        print(f"\n[DONE] Episode finished! Cumulative Reward: {total_reward:.2f}")
        break

env.close()