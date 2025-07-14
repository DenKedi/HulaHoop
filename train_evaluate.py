# train_evaluate.py

import os
import time
import json
import csv
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter
import time
from hulahoop_env import HulaHoopEnv


class CsvLoggingCallback(BaseCallback):
    def __init__(self, csv_file_path, verbose=0):
        super(CsvLoggingCallback, self).__init__(verbose)
        self.csv_file_path = csv_file_path
        self.csv_file = None
        self.csv_writer = None

    def _on_training_start(self) -> None:
        self.csv_file = open(self.csv_file_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestep", "episode_reward", "episode_length"])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        for info in self.model.ep_info_buffer:
            self.csv_writer.writerow([self.num_timesteps, info['r'], info['l']])

    def _on_training_end(self) -> None:
        if self.csv_file:
            self.csv_file.close()

# Definition der Experimente
BASELINE_CONFIG = {
    "model_class": DQN, "learning_rate": 5e-5, "buffer_size": 200000,
    "learning_starts": 10000, "exploration_fraction": 0.2, "exploration_initial_eps": 0.7,
    "exploration_final_eps": 0.02,
}

EXPERIMENTS = []
for lr in [1e-4, 5e-5]:
    config = BASELINE_CONFIG.copy();
    config["learning_rate"] = lr;
    config["name"] = f"DQN_lr_{lr}";
    EXPERIMENTS.append(config)
for frac in [0.2, 0.4, 0.6, 0.8, 1]:
    config = BASELINE_CONFIG.copy();
    config["exploration_fraction"] = frac;
    config["name"] = f"DQN_expl_frac_{frac}";
    EXPERIMENTS.append(config)
for initial_eps in [0.5, 0.7, 1.0]:
    config = BASELINE_CONFIG.copy();
    config["exploration_initial_eps"] = initial_eps;
    config["name"] = f"DQN_init_eps_{initial_eps}";
    EXPERIMENTS.append(config)
for final_eps in [0.02, 0.05, 0.1]:
    config = BASELINE_CONFIG.copy();
    config["exploration_final_eps"] = final_eps;
    config["name"] = f"DQN_final_eps_{final_eps}";
    EXPERIMENTS.append(config)
for buffer in [50000, 100000, 200000]:
    config = BASELINE_CONFIG.copy();
    config["buffer_size"] = buffer;
    config["name"] = f"DQN_buffer_{buffer}";
    EXPERIMENTS.append(config)
for starts in [2000, 5000, 10000]:
    config = BASELINE_CONFIG.copy();
    config["learning_starts"] = starts;
    config["name"] = f"DQN_starts_{starts}";
    EXPERIMENTS.append(config)


if __name__ == '__main__':
    num_runs_per_experiment = 5
    total_timesteps_per_run = 170000
    n_eval_episodes = 30
    results_parent_dir = "experiment_results"
    os.makedirs(results_parent_dir, exist_ok=True)

    for config in EXPERIMENTS:
        experiment_name = f"{config['name']}_{int(time.time())}"
        experiment_dir = os.path.join(results_parent_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print("\n" + "=" * 80);
        print(f"STARTE EXPERIMENT: {config['name']}");
        print("=" * 80)
        run_results = []

        for i in range(num_runs_per_experiment):
            run_name = f"run_{i + 1}"
            run_dir = os.path.join(experiment_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            print(f"\n--- Starte Durchlauf {i + 1}/{num_runs_per_experiment} für {config['name']} ---")

            env = HulaHoopEnv()
            log_path = os.path.join(run_dir, "training_log.csv")
            callback = CsvLoggingCallback(csv_file_path=log_path)

            model = config["model_class"](
                "MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), verbose=1,
                learning_rate=config["learning_rate"], gamma=0.99,
                buffer_size=config["buffer_size"], learning_starts=config["learning_starts"],
                exploration_fraction=config["exploration_fraction"],
                exploration_initial_eps=config["exploration_initial_eps"],
                exploration_final_eps=config["exploration_final_eps"],
                batch_size=32, train_freq=4, gradient_steps=1, seed=i,
                tensorboard_log=os.path.join(experiment_dir, "tensorboard_logs")
            )

            model.learn(total_timesteps=total_timesteps_per_run, callback=callback, progress_bar=True)

            print(f"--- Starte Evaluierung für Durchlauf {i + 1} ---")
            eval_env = HulaHoopEnv()


            episode_details = []
            max_eval_steps_per_episode = 5000  # Timeout für perfekte Läufe

            for ep in range(n_eval_episodes):
                obs, info = eval_env.reset()
                done = False
                truncated = False
                episode_reward = 0
                episode_length = 0
                while not done and not truncated:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    if episode_length >= max_eval_steps_per_episode:
                        truncated = True

                reason = "Timeout" if truncated else info.get("termination_reason", "Unbekannt")
                episode_details.append({
                    "episode": ep + 1,
                    "reward": episode_reward,
                    "length": episode_length,
                    "termination_reason": reason
                })

            # Speichere die detaillierten Ergebnisse in einer CSV-Datei
            eval_log_path = os.path.join(run_dir, "evaluation_details.csv")
            with open(eval_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=episode_details[0].keys())
                writer.writeheader()
                writer.writerows(episode_details)

            run_rewards = [d['reward'] for d in episode_details]
            run_results.append(run_rewards)

            model.save(os.path.join(run_dir, "final_model.zip"))

            env.close()
            eval_env.close()
            print(f"Durchlauf {i + 1} beendet. Avg. Reward: {np.mean(run_rewards):.2f} +/- {np.std(run_rewards):.2f}")

            flat_rewards = [reward for run in run_results for reward in run]
            summary = {
                "experiment_config": {k: v for k, v in config.items() if k != 'model_class'},
                "evaluation_summary": {
                    "mean_reward_over_all_runs": np.mean(flat_rewards),
                    "std_reward_over_all_runs": np.std(flat_rewards),
                    "mean_reward_per_run": [np.mean(run) for run in run_results]
                }
            }
        report_path = os.path.join(experiment_dir, "summary_report.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4, default=str)

        print("\n" + "-" * 80);
        print(f"EXPERIMENT '{config['name']}' ABGESCHLOSSEN.")
        print(f"Gesamtdurchschnittliche Belohnung: {summary['evaluation_summary']['mean_reward_over_all_runs']:.2f}")
        print(f"Standardabweichung über alle Episoden: {summary['evaluation_summary']['std_reward_over_all_runs']:.2f}")
        print(f"Detaillierter Bericht und Logs gespeichert in: {experiment_dir}")
        print("-" * 80)