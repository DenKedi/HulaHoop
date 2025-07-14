import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE_PATH = "./hula_dqn_tensorboard/loss_log.csv"

SMOOTHING_WINDOW = 100


try:
    data = pd.read_csv(LOG_FILE_PATH)
    print(f"Daten erfolgreich aus '{LOG_FILE_PATH}' geladen.")
    print(f"Insgesamt {len(data)} Datenpunkte gefunden.")

    data['loss_smoothed'] = data['loss'].rolling(window=SMOOTHING_WINDOW).mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(data['timesteps'], data['loss'], color='lightcoral', alpha=0.4, label='Tatsächlicher Loss')

    ax.plot(data['timesteps'], data['loss_smoothed'], color='firebrick', linewidth=2,
            label=f'Geglätteter Loss (Fenster={SMOOTHING_WINDOW})')

    ax.set_title('Überschätzung der Q-Values', fontsize=16)
    ax.set_xlabel('Anzahl der Timesteps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale('log')

    print("\nPlot-Fenster wird angezeigt...")

    plt.show()

except FileNotFoundError:
    print(f"FEHLER: Die Datei '{LOG_FILE_PATH}' wurde nicht gefunden.")
    print("Stelle sicher, dass der Pfad korrekt ist und das Trainings-Skript bereits gelaufen ist.")

except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")