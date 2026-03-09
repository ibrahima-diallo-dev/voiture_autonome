"""Creation d'un environnement Highway avec une configuration adaptee au projet."""

import gymnasium as gym


def create_env(env_name):
    """Construit un environnement Gymnasium configure pour l'apprentissage DQN."""
    # Configuration metier: observation, action et fonction de recompense.
    config = {
        "observation": {
            "type": "Kinematics", # La voiture voit les distances et vitesses
            "vehicles_count": 5,  # Elle surveille les 5 voitures les plus proches
        },
        "action": {
            "type": "DiscreteMetaAction", # [Gaucher, Rien, Droite, Accélérer, Freiner]
            "duration": 180, # Chaque action dure 3 secondes (180 frames à 60fps)
        },
        "collision_reward": -15,   # ON AUGMENTE LA PUNITION (défaut souvent à -1)
        "high_speed_reward": 0.5,  # On récompense un peu moins la vitesse pure
        "lane_change_reward": 0.2, # ON AJOUTE UN BONUS pour l'inciter à changer de voie
        "reward_speed_range": [20, 30],
        "offroad_terminal": True   # Si elle sort de la route, c'est fini
    }

    env = gym.make(env_name, config=config)
    return env