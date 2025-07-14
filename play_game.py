import pygame
import time
from stable_baselines3 import DQN
from train_agent import HulaHoopEnv


# --- Interaktions-Parameter ---

NUDGE_DOWN_POWER = 2.5
NUDGE_UP_POWER = -2.5

# Schwellenwert, ab dem ein Eingriff möglich ist.
INTERVENTION_THRESHOLD_STEPS = 20

if __name__ == '__main__':
    env = HulaHoopEnv(render_mode="human")

    try:
        model = DQN.load("hula_dqn_model", env=env)
        print("Modell 'hula_dqn_model.zip' erfolgreich geladen.")
        print("\n--- STEUERUNG ---")
        print("Linksklick: Reifen sanft nach UNTEN stupsen.")
        print("Rechtsklick: Reifen sanft nach OBEN stupsen.")
        print("Eingriff erst nach 20Schritten möglich.")
        print("-----------------\n")

    except FileNotFoundError:
        print("Fehler: Die Datei 'hula_dqn_model.zip' wurde nicht gefunden.")
        exit()

    while True:
        obs, info = env.reset()
        done = False
        episode_steps = 0

        print(f"\n--- Starte neue Episode ---")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if episode_steps > INTERVENTION_THRESHOLD_STEPS:

                        if event.button == 1:  # Linksklick
                            print(f"Eingriff bei Schritt {episode_steps}: Stupse nach UNTEN.")
                            env.hoop_y_velocity += NUDGE_DOWN_POWER
                        elif event.button == 3:  # Rechtsklick
                            print(f"Eingriff bei Schritt {episode_steps}: Stupse nach OBEN.")
                            env.hoop_y_velocity += NUDGE_UP_POWER
                    else:
                        print(f"Noch {INTERVENTION_THRESHOLD_STEPS - episode_steps} Schritte bis Eingriff möglich.")


            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_steps += 1
            time.sleep(1 / 60)

        # Episode ist beendet
        reason = info.get("termination_reason", "Unbekannt")
        print(f"Episode nach {episode_steps} Schritten beendet. Grund: {reason}")
        time.sleep(1)