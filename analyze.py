import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from collections import defaultdict

def extract_group_key(name):
    match = re.match(r"DQN_([a-zA-Z_]+)_", name)
    return match.group(1) if match else "ungrouped"

def extract_group_value(name):
    match = re.match(r"DQN_[a-zA-Z_]+_(.+)", name)
    return match.group(1) if match else "unknown"

def analyze_grouped_experiments(parent_dir):
    grouped_data = defaultdict(list)
    print(f"🔍 Durchsuche Verzeichnis: {parent_dir}")

    for experiment_name in os.listdir(parent_dir):
        experiment_path = os.path.join(parent_dir, experiment_name)
        if not os.path.isdir(experiment_path):
            continue

        group_key = extract_group_key(experiment_name)
        group_value = extract_group_value(experiment_name)
        print(f"📁 Lade Experiment: {experiment_name} → Gruppe: {group_key} = {group_value}")

        run_dfs = []
        for run in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run, "training_log.csv")
            if os.path.exists(run_path):
                df = pd.read_csv(run_path)
                run_dfs.append(df)

        if not run_dfs:
            print("⚠️  Keine gültigen Runs gefunden.")
            continue

        master_timesteps = np.linspace(0, 150000, num=500)
        interpolated = [
            np.interp(master_timesteps, df['timestep'], df['episode_reward'])
            for df in run_dfs
        ]

        mean_reward = np.mean(interpolated, axis=0)
        grouped_data[(group_key, group_value)].append((master_timesteps, mean_reward))

    return grouped_data

def plot_grouped_results(grouped_data):
    summary_rows = []
    all_groups = defaultdict(list)

    for (group_key, group_value), curves in grouped_data.items():
        all_groups[group_key].append((group_value, curves))

    for group_key, variants in all_groups.items():
        print(f"\n📊 Erzeuge Plot für Hyperparameter: {group_key}")
        plt.figure(figsize=(12, 7))
        plt.title(f"Vergleich: {group_key}", fontsize=16, weight='bold')

        try:
            variants_sorted = sorted(variants, key=lambda x: float(x[0].replace("e-", "E-").replace("e", "E")))
        except ValueError:
            variants_sorted = sorted(variants, key=lambda x: x[0])  # fallback

        for group_value, curves in variants_sorted:
            mean_curve = np.mean([c[1] for c in curves], axis=0)
            std_curve = np.std([c[1] for c in curves], axis=0)
            timesteps = curves[0][0]

            print(f"  ➤ {group_value}: Final Avg = {np.mean(mean_curve[-50:]):.2f}")
            plt.plot(timesteps, mean_curve, label=f"{group_value}")
            plt.fill_between(timesteps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

            summary_rows.append({
                "Hyperparameter": group_key,
                "Wert": group_value,
                "Avg Final Reward": round(np.mean(mean_curve[-50:]), 2)
            })

        plt.xlabel("Timesteps")
        plt.ylabel("Durchschnittsreward")
        plt.grid(True)
        plt.legend(title=group_key)
        plt.tight_layout()
        plt.savefig(f"grouped_plot_{group_key}.png")
        plt.close()
        print(f"✅ Grafik gespeichert: grouped_plot_{group_key}.png")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by=["Hyperparameter", "Wert"], inplace=True)
    df_summary.to_csv("summary_table.csv", index=False)
    print("\n📄 Zusammenfassung gespeichert: summary_table.csv")
    return df_summary

if __name__ == '__main__':
    base_path = "experiment_results"

    if not os.path.exists(base_path):
        print("❌ Fehler: experiment_results Verzeichnis nicht gefunden.")
        exit()

    grouped = analyze_grouped_experiments(base_path)
    df = plot_grouped_results(grouped)

    print("\n📈 Fertig. Ergebnisse:")
    print(df)
