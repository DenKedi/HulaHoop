import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import Counter
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
# Import für den Callback und zum Speichern von CSV-Dateien
from stable_baselines3.common.callbacks import BaseCallback
import os


class HulaHoopEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # Aktion: 0 (nichts tun) oder 1 (Schwung geben)
        self.action_space = spaces.Discrete(2)

        # Beobachtung: Ein Vektor mit 3 Werten:
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Zustandsvariablen
        self.hoop_y = 0.0
        self.hoop_y_velocity = 0.0
        self.swing_speed = 0.0

        # HUD-Anzeige
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.last_action = 0
        self.last_reward = 0.0

        # SPIELCODE
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

        # Pygame-spezifische Variablen
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

        # Startposition des Reifens (+/- 20 Pixel um die Mitte)
        sweet_spot_center = self.robot_y + self.robot_height / 2
        self.hoop_y = sweet_spot_center + self.np_random.uniform(-20, 20)

        # Start-Schwunggeschwindigkeit leicht variieren
        self.swing_speed = self.np_random.uniform(7.0, 10.0)

        self.hoop_y_velocity = 0.0
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):

        # --- Aktion ausführen ---
        if action == 1:
            self.swing_speed += self.SWING_BOOST
            self.hoop_y_velocity -= self.KICK_POWER

        # --- Physik-Update ---
        self.swing_speed *= self.SWING_DECAY
        lift_force = self.swing_speed * self.LIFT_FACTOR
        if self.swing_speed > self.SWEET_SPOT_MAX:
            lift_force += (self.swing_speed - self.SWEET_SPOT_MAX) * self.UPWARD_FLIGHT_FACTOR
        self.hoop_y_velocity += self.GRAVITY - lift_force
        self.hoop_y += self.hoop_y_velocity

        # --- Zustand prüfen und Belohnung (Reward) definieren ---
        terminated = self.hoop_y > self.GAME_OVER_BOTTOM or self.hoop_y < self.GAME_OVER_TOP
        info = self._get_info()

        # Reward-Logik
        reward = 0
        if terminated:
            reward = -100  # Große Strafe für's Verlieren
            if self.hoop_y > self.GAME_OVER_BOTTOM:
                info["termination_reason"] = "Gefallen"
            else:
                info["termination_reason"] = "Hochgeflogen"
        else:
            # Belohnung dafür, in der Nähe des Zentrums zu sein
            sweet_spot_center = self.robot_y + self.robot_height / 2
            distance_to_center = abs(self.hoop_y - sweet_spot_center)

            # Die Belohnung ist höher, je näher der Reifen am Zentrum ist.
            max_distance = 80
            if distance_to_center < max_distance:
                # Linearer Bonus: 2.0 wenn perfekt im Zentrum, 0 am Rand der Zone.
                reward = 2.0 * (1 - (distance_to_center / max_distance))

            # Bestrafung für hohe Geschwindigkeit
            reward -= abs(self.hoop_y_velocity) * 0.1

        self.last_action = action
        self.last_reward = reward

        self.current_episode_steps += 1
        self.current_episode_reward += reward

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

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
        sweet_spot_color = (0, 255, 0)  # Grün

        top_y = sweet_spot_center - max_distance
        bottom_y = sweet_spot_center + max_distance

        pygame.draw.line(self.screen, sweet_spot_color, (0, top_y), (self.SCREEN_WIDTH, top_y), 2)
        pygame.draw.line(self.screen, sweet_spot_color, (0, bottom_y), (self.SCREEN_WIDTH, bottom_y), 2)

        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_rect)
        pygame.draw.rect(self.screen, self.ROBOT_COLOR, self.robot_head_rect)
        hoop_width = 150 + (self.hoop_y - self.robot_y) * 0.1
        hoop_rect = pygame.Rect(self.robot_x + self.robot_width / 2 - hoop_width / 2, self.hoop_y, hoop_width, 25)
        pygame.draw.ellipse(self.screen, self.HOOP_COLOR, hoop_rect, 6)

        hud_texts = [
            "--- AGENTEN-STATUS ---",
            f"Schwung: {self.swing_speed:.2f}",
            f"Position Y: {self.hoop_y:.2f}",
            f"Geschwindigkeit Y: {self.hoop_y_velocity:.2f}",
            "--- AKTION & BELOHNUNG ---",
            f"Letzte Aktion: {'Schwung geben' if self.last_action == 1 else 'Nichts tun'}",
            f"Frame-Belohnung: {self.last_reward:.2f}",
            "--- EPISODEN-STATUS ---",
            f"Schritte: {self.current_episode_steps}",
            f"Gesamtbelohnung: {self.current_episode_reward:.2f}",
        ]

        for i, text in enumerate(hud_texts):
            text_surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (10, 10 + i * 25))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


# Eigener Callback, um den Loss zu protokollieren
class LossLoggingCallback(BaseCallback):
    def __init__(self, log_path: str, verbose=0):
        super(LossLoggingCallback, self).__init__(verbose)
        self.log_path = log_path
        self.file_handler = None

    def _on_training_start(self):
        # Erstellt den Ordner, falls er nicht existiert
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.file_handler = open(self.log_path, "w")
        self.file_handler.write("timesteps,loss\n")

    def _on_step(self) -> bool:
        if 'train/loss' in self.model.logger.name_to_value:
            loss = self.model.logger.name_to_value['train/loss']
            timesteps = self.num_timesteps
            self.file_handler.write(f"{timesteps},{loss}\n")
        return True

    def _on_training_end(self):
        if self.file_handler is not None:
            self.file_handler.close()


if __name__ == '__main__':

    train_env = HulaHoopEnv()
    check_env(train_env)

    policy_kwargs = dict(
        net_arch=[256, 256]
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=200000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        learning_rate=5e-5,
        tensorboard_log="./hula_dqn_tensorboard/",
        exploration_fraction=0.2,
        exploration_initial_eps=0.7,
        exploration_final_eps=0.02
    )

    loss_log_path = "./hula_dqn_tensorboard/loss_log.csv"
    loss_callback = LossLoggingCallback(log_path=loss_log_path)

    print("Beginne mit dem Training...")
    model.learn(total_timesteps=170000, progress_bar=True, callback=loss_callback)

    model.save("hula_dqn_model")
    print(f"Training abgeschlossen. Modell gespeichert und Loss-Werte in '{loss_log_path}' protokolliert.")

    train_env.close()

    # EVALUIERUNG DES TRAINIERTEN MODELLS
    print("\n" + "=" * 50)
    print("--- STARTE FINALE EVALUIERUNG DES MODELLS ---")
    print("=" * 50)

    model = DQN.load("hula_dqn_model")

    eval_env = HulaHoopEnv(render_mode="human")

    n_eval_episodes = 20
    max_steps_per_episode = 3000
    termination_reasons = Counter()
    total_rewards = []

    for episode in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_steps += 1

            if episode_steps >= max_steps_per_episode:
                break
        if episode_steps >= max_steps_per_episode:
            reason = "Timeout (Perfekt gespielt)"
        else:
            reason = info.get("termination_reason", "Unbekannt")
        termination_reasons[reason] += 1
        total_rewards.append(episode_reward)
        print(
            f"Eval-Episode {episode + 1}/{n_eval_episodes} | Schritte: {episode_steps} | Belohnung: {episode_reward:.2f} | Grund: {reason}")

    # Ausgabe
    print("\n--- EVALUIERUNGS-ZUSAMMENFASSUNG ---")
    print(f"Durchschnittliche Belohnung über {n_eval_episodes} Episoden: {np.mean(total_rewards):.2f}")
    print("Statistik der End-Gründe:")
    for reason, count in termination_reasons.items():
        print(f"- {reason}: {count} mal ({(count / n_eval_episodes) * 100:.1f}%)")
    print("=" * 50)

    eval_env.close()