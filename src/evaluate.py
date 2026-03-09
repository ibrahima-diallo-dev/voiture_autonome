"""Evaluation visuelle d'un modele DQN entraine sur highway."""

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from src.config import ENV_NAME, MODEL_SAVE_PATH
import time


def evaluate():
    """Joue plusieurs episodes en mode rendu humain et affiche les scores."""
    # 1. Créer l'environnement en mode visuel
    env = gym.make(ENV_NAME, render_mode="human")
    
    # 2. Configurer la durée (ex: 60 secondes pour mieux voir)
    env.unwrapped.configure({"duration": 60})
    
    # 3. Charger ton modèle expert
    print(f"Chargement du modèle : {MODEL_SAVE_PATH}")
    model = DQN.load(MODEL_SAVE_PATH)
    
    # Lancer 5 tests d'affilee pour observer un comportement moyen.
    for episode in range(1, 11):
        obs, info = env.reset()
        done = truncated = False
        score = 0
        
        print(f"\n--- Épisode {episode} en cours ---")
        
        while not (done or truncated):
            # L'IA choisit l'action de manière déterministe (la meilleure option)
            action, _states = model.predict(obs, deterministic=True)
            
            # On applique l'action
            obs, reward, done, truncated, info = env.step(action)
            score += reward
            
            # On affiche le rendu
            env.render()
            
            # Petit délai pour que l'œil humain puisse suivre (facultatif)
            # time.sleep(0.02)

        if done:
            print(f"Fin de l'épisode {episode} : COLLISION ! (Score: {score:.2f})")
        else:
            print(f"Fin de l'épisode {episode} : TEMPS ÉCOULÉ (Score: {score:.2f})")
        
        time.sleep(1) # Pause de 1s entre chaque essai

    env.close()

if __name__ == "__main__":
    evaluate()