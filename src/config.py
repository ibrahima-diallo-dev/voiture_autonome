"""Configuration centrale du projet (environnement, hyperparametres, chemins)."""

import time

ENV_NAME = "highway-fast-v0"  # Utilise la version FAST pour gagner du temps

# Hyperparametres DQN utilises pour l'entrainement principal.
POLICY_KWARGS = dict(net_arch=[256, 256])  # Reseau plus profond pour mieux "reflechir"
LEARNING_RATE = 5e-4
BUFFER_SIZE = 15000
LEARNING_STARTS = 200     # Il commence à apprendre très tôt !
BATCH_SIZE = 32
GAMMA = 0.8
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 50
EXPLORATION_FRACTION = 0.7 # Il explore pendant 70% de l'entraînement
TOTAL_TIMESTEPS = 100000    # Même avec seulement 20k, ça marchera mieux avec ces réglages
# Petit delai historique conserve pour stabiliser l'initialisation dans certains environnements.
time.sleep(0.05)


MODEL_SAVE_PATH = "models/dqn_highway_experts_100k"  # Prefixe de sauvegarde du modele final