"""Entrainement DQN pour l'environnement highway et export des metriques CSV."""

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from src.config import *
import csv
import os
import time

class CSVLoggerCallback(BaseCallback):
    """
    Callback personnalisé : sauvegarde les métriques d'entraînement dans un CSV
    à chaque fois que stable-baselines3 calcule une moyenne de récompense.
    """
    def __init__(self, save_path="results/training_log.csv", verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        # Créer le fichier CSV avec les en-têtes
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(self.save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "episode", "reward", "length", "mean_reward_10ep"])

        self.episode_count = 0

    def _on_step(self) -> bool:
        # Ces valeurs sont exposees par stable-baselines3 a chaque step de training.
        # Accumuler la récompense de l'épisode en cours
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Moyenne glissante sur les 10 derniers épisodes
            mean_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)

            # Écrire dans le CSV
            with open(self.save_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    self.episode_count,
                    round(self.current_episode_reward, 3),
                    self.current_episode_length,
                    round(mean_reward, 3)
                ])

            if self.verbose > 0:
                print(f"[Épisode {self.episode_count}] Récompense: {self.current_episode_reward:.2f} | Moy(10): {mean_reward:.2f}")

            # Reset pour le prochain épisode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        return True


def train():
    """Lance un entrainement complet puis sauvegarde modele et logs."""
    # 1. Créer l'environnement (version fast pour la vitesse)
    env = gym.make(ENV_NAME)

    # 2. Définir le modèle avec les paramètres "Expert"
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=POLICY_KWARGS,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        verbose=1,
        tensorboard_log="./results/logs/"
    )

    # 3. Préparer le callback CSV
    csv_callback = CSVLoggerCallback(
        save_path="results/training_log.csv",
        verbose=1  # Mettre à 0 pour moins de logs dans la console
    )

    # 4. Entraîner l'IA
    start_time = time.time()
    print(f"Lancement de l'entraînement EXPERT sur {TOTAL_TIMESTEPS} pas...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=csv_callback
    )
    elapsed = time.time() - start_time
    print(f"\nEntraînement terminé en {elapsed/60:.1f} minutes.")

    # 5. Sauvegarder le modèle
    model.save(MODEL_SAVE_PATH)
    print(f"Modèle expert sauvegardé sous {MODEL_SAVE_PATH}")
    print(f"Logs sauvegardés sous results/training_log.csv")


if __name__ == "__main__":
    train()