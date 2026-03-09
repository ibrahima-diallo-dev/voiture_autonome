"""Visualisation des metriques d'entrainement exportees dans results/training_log.csv."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

LOG_PATH = "results/training_log.csv"
OUTPUT_DIR = "results"


def load_logs(path=LOG_PATH):
    """Charge le CSV de logs et verifie sa presence."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Lance d'abord train_dqn.py pour générer les logs."
        )
    df = pd.read_csv(path)
    print(f"{len(df)} épisodes chargés depuis {path}")
    return df


def plot_reward_curve(df, save=True):
    """Courbe de récompense par épisode + moyenne glissante sur 10 épisodes."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["episode"], df["reward"], color="#b0c4de", linewidth=0.8,
            alpha=0.7, label="Récompense par épisode")
    ax.plot(df["episode"], df["mean_reward_10ep"], color="#1a3a6b", linewidth=2,
            label="Moyenne glissante (10 épisodes)")

    ax.set_title("Courbe d'apprentissage — DQN Highway", fontsize=14, fontweight="bold")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense cumulée")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "reward_plot.png")
        plt.savefig(path, dpi=150)
        print(f"Graphique sauvegardé : {path}")
    plt.show()


def plot_episode_lengths(df, save=True):
    """Durée des épisodes en nombre de pas — indique la survie de l'agent."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(df["episode"], df["length"], color="#f0c040", alpha=0.8, width=0.8)
    ax.plot(df["episode"],
            df["length"].rolling(10).mean(),
            color="#1a3a6b", linewidth=2, label="Moyenne mobile (10 épisodes)")

    ax.set_title("Durée des épisodes (nombre de pas)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Longueur (pas)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "episode_length_plot.png")
        plt.savefig(path, dpi=150)
        print(f"Graphique sauvegardé : {path}")
    plt.show()


def plot_timestep_reward(df, save=True):
    """Récompense moyenne en fonction du nombre de timesteps (vue globale)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["timestep"], df["mean_reward_10ep"], color="#2ecc71", linewidth=2)
    ax.fill_between(df["timestep"], df["mean_reward_10ep"],
                    alpha=0.15, color="#2ecc71")

    ax.set_title("Performance au fil du temps (timesteps)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Récompense moyenne (10 épisodes)")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "timestep_reward_plot.png")
        plt.savefig(path, dpi=150)
        print(f"Graphique sauvegardé : {path}")
    plt.show()


def print_summary(df):
    """Affiche un résumé statistique dans la console."""
    print("\n" + "="*45)
    print("        RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*45)
    print(f"  Nombre total d'épisodes  : {len(df)}")
    print(f"  Timesteps total          : {df['timestep'].max():,}")
    print(f"  Récompense max           : {df['reward'].max():.2f}")
    print(f"  Récompense min           : {df['reward'].min():.2f}")
    print(f"  Récompense moyenne       : {df['reward'].mean():.2f}")
    print(f"  Moy. 10 derniers épisodes: {df['mean_reward_10ep'].iloc[-1]:.2f}")
    print(f"  Durée moy. épisode (pas) : {df['length'].mean():.0f}")
    print("="*45 + "\n")


if __name__ == "__main__":
    # Point d'entree utilitaire: resume + generation de tous les graphiques.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_logs()
    print_summary(df)
    plot_reward_curve(df)
    plot_episode_lengths(df)
    plot_timestep_reward(df)
    print("Tous les graphiques ont été générés dans le dossier results/")