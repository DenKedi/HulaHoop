# hula_hoop_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class HulaHoopEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.hoop_y = 0.0
        self.hoop_y_velocity = 0.0
        self.swing_speed = 0.0
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.last_action = 0
        self.last_reward = 0.0
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.BACKGROUND_COLOR = (20, 30, 40)
        self.WHITE = (255, 255, 255)
        self.ROBOT_COLOR = (130, 140, 150)
        self.HOOP_COLOR = (255, 105, 180)
        self.robot_width = 80
        self.robot_height = 200
        self.robot_x = (self.SCREEN_WIDTH - self.robot_width) / 2
        self.robot_y = self.SCREEN_HEIGHT - self.robot_height - 20
        self.robot_rect = pygame.Rect(self.robot_x, self.robot_y, self.robot_width, self.robot_height)
        self.robot_head_rect = pygame.Rect(self.robot_x - 10, self.robot_y - 40, self.robot_width + 20, 40)
        self.SWING_DECAY = 0.99
        self.SWING_BOOST = 2.0
        self.GRAVITY = 0.15
        self.KICK_POWER = 3.5
        self.LIFT_FACTOR = 0.01
        self.UPWARD_FLIGHT_FACTOR = 0.04
        self.SWEET_SPOT_MIN = 12.0
        self.SWEET_SPOT_MAX = 22.0
        self.GAME_OVER_BOTTOM = self.SCREEN_HEIGHT - 30
        self.GAME_OVER_TOP = self.robot_y - 20
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        return np.array([self.hoop_y, self.hoop_y_velocity, self.swing_speed], dtype=np.float32)

    def _get_info(self):
        return {"swing_speed": self.swing_speed}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sweet_spot_center = self.robot_y + self.robot_height / 2
        self.hoop_y = sweet_spot_center + self.np_random.uniform(-20, 20)
        self.swing_speed = self.np_random.uniform(7.0, 10.0)
        self.hoop_y_velocity = 0.0
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        if self.render_mode == "human": self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action == 1:
            self.swing_speed += self.SWING_BOOST
            self.hoop_y_velocity -= self.KICK_POWER
        self.swing_speed *= self.SWING_DECAY
        lift_force = self.swing_speed * self.LIFT_FACTOR
        if self.swing_speed > self.SWEET_SPOT_MAX:
            lift_force += (self.swing_speed - self.SWEET_SPOT_MAX) * self.UPWARD_FLIGHT_FACTOR
        self.hoop_y_velocity += self.GRAVITY - lift_force
        self.hoop_y += self.hoop_y_velocity
        terminated = self.hoop_y > self.GAME_OVER_BOTTOM or self.hoop_y < self.GAME_OVER_TOP
        info = self._get_info()
        reward = 0
        if terminated:
            reward = -100
            if self.hoop_y > self.GAME_OVER_BOTTOM:
                info["termination_reason"] = "Gefallen"
            else:
                info["termination_reason"] = "Hochgeflogen"
        else:
            sweet_spot_center = self.robot_y + self.robot_height / 2
            distance_to_center = abs(self.hoop_y - sweet_spot_center)
            max_distance = 80
            if distance_to_center < max_distance:
                reward = 2.0 * (1 - (distance_to_center / max_distance))
            reward -= abs(self.hoop_y_velocity) * 0.1
        self.last_action = action
        self.last_reward = reward
        self.current_episode_steps += 1
        self.current_episode_reward += reward
        if self.render_mode == "human": self._render_frame()
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "human": return self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Hula Hoop Roboter - KI Interaktion")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 22)

        self.screen.fill(self.BACKGROUND_COLOR)

        max_distance = 80
        sweet_spot_center = self.robot_y + self.robot_height / 2

        top_y = sweet_spot_center - max_distance
        bottom_y = sweet_spot_center + max_distance

        sweet_spot_height = bottom_y - top_y
        sweet_spot_surface = pygame.Surface((self.SCREEN_WIDTH, sweet_spot_height), pygame.SRCALPHA)
        sweet_spot_surface.fill((0, 255, 100, 60))

        self.screen.blit(sweet_spot_surface, (0, top_y))

        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_rect)
        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_head_rect)

        hoop_width = 150 + (self.hoop_y - self.robot_y) * 0.1
        hoop_rect = pygame.Rect(self.robot_x + self.robot_width / 2 - hoop_width / 2, self.hoop_y, hoop_width, 25)
        pygame.draw.ellipse(self.screen, self.HOOP_COLOR, hoop_rect, 6)

        hud_texts = ["--- AGENTEN-STATUS ---", f"Schwung: {self.swing_speed:.2f}", f"Position Y: {self.hoop_y:.2f}",
                     f"Geschwindigkeit Y: {self.hoop_y_velocity:.2f}", "--- AKTION & BELOHNUNG ---",
                     f"Letzte Aktion: {'Schwung geben' if self.last_action == 1 else 'Nichts tun'}",
                     f"Frame-Belohnung: {self.last_reward:.2f}", "--- EPISODEN-STATUS ---",
                     f"Schritte: {self.current_episode_steps}", f"Gesamtbelohnung: {self.current_episode_reward:.2f}"]

        for i, text in enumerate(hud_texts):
            text_surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (10, 10 + i * 20))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()