<p align="center">
	<img src="https://capsule-render.vercel.app/api?type=rect&height=240&color=0:111111,25:ef4444,50:f97316,75:facc15,100:111111&text=VOITURE%20AUTONOME%20%7C%20DQN&fontColor=ffffff&fontSize=46&fontAlignY=40&desc=RL%20sur%20highway-fast-v0%20%7C%20Mode%20Racing&descAlignY=64&descSize=16" alt="Voiture Autonome DQN Racing Banner" />
</p>

<p align="center">
	<img src="https://img.shields.io/badge/Python-3.13-1f2937?style=for-the-badge&logo=python&logoColor=facc15" alt="Python" />
	<img src="https://img.shields.io/badge/Gymnasium-1.2.3-991b1b?style=for-the-badge" alt="Gymnasium" />
	<img src="https://img.shields.io/badge/highway--env-Simulator-7f1d1d?style=for-the-badge" alt="highway-env" />
	<img src="https://img.shields.io/badge/Stable--Baselines3-DQN-c2410c?style=for-the-badge" alt="Stable-Baselines3" />
	<img src="https://img.shields.io/badge/Track-Training%20Ready-111827?style=for-the-badge" alt="Track Status" />
</p>

## Apercu

Ce projet entraine un agent de conduite autonome avec **DQN** dans `highway-fast-v0`.

Mission:

- survivre plus longtemps,
- eviter les collisions,
- ameliorer la recompense moyenne episode apres episode.

Le flux de travail est deja complet:

- entrainement,
- logging CSV,
- evaluation visuelle,
- visualisation des performances.

---

## Pit Stop Rapide

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash
python -m pip install -r requirements.txt
python -m src.train_dqn
python -m src.evaluate
python -m src.plot_results
```

---

## Stack

- `gymnasium`
- `highway-env`
- `stable-baselines3[extra]`
- `shimmy`
- `matplotlib`
- `pandas`

---

## Arborescence

```text
voiture_autonome/
|- data/
|- logs/
|- models/
|- notebooks/
|- results/
|- src/
|  |- config.py         # Hyperparametres et chemins
|  |- train_dqn.py      # Entrainement + callback CSV
|  |- evaluate.py       # Evaluation visuelle (render human)
|  |- plot_results.py   # Courbes et resume statistique
|  |- env_wrapper.py    # Config personnalisee de l'environnement
|- requirements.txt
|- README.md
```

---

## Installation detaillee

### 1. Cloner

```bash
git clone <URL_DU_REPO>
cd voiture_autonome
```

### 2. Environnement virtuel

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Dependances

```bash
python -m pip install -r requirements.txt
```

---

## Commandes de course

### Lancer l'entrainement

```bash
python -m src.train_dqn
```

Sorties:

- modele: `models/dqn_highway_experts_100k`
- logs: `results/training_log.csv`

### Lancer l'evaluation visuelle

```bash
python -m src.evaluate
```

### Generer les graphes

```bash
python -m src.plot_results
```

Graphiques produits dans `results/`:

- `reward_plot.png`
- `episode_length_plot.png`
- `timestep_reward_plot.png`

---

## Reglages DQN actuels

Depuis `src/config.py`:

- `ENV_NAME=highway-fast-v0`
- `net_arch=[256, 256]`
- `LEARNING_RATE=5e-4`
- `BUFFER_SIZE=15000`
- `GAMMA=0.8`
- `EXPLORATION_FRACTION=0.7`
- `TOTAL_TIMESTEPS=100000`

---

## Resultats

Tu peux embed les images dans GitHub:

```md
![Reward Curve](results/reward_plot.png)
![Episode Length](results/episode_length_plot.png)
![Timestep Reward](results/timestep_reward_plot.png)
```

---

## Notebooks

Le dossier `notebooks/` contient:

- `notebooks/01_test_environnement.ipynb`
- `notebooks/02_exploration_donnees.ipynb`

---

## Conseils

- Utiliser l'interpreteur `./.venv/Scripts/python.exe` dans VS Code.
- Si un import reste en erreur: reload de la fenetre VS Code.
- Relancer `python -m src.plot_results` apres chaque session d'entrainement.

---

## Roadmap

- [ ] sauvegarde automatique du meilleur checkpoint
- [ ] comparaison DQN vs PPO
- [ ] variation des fonctions de recompense
- [ ] export video des episodes

---

<p align="center">
	<img src="https://img.shields.io/badge/Racing%20Mode-ON-b91c1c?style=for-the-badge" alt="Racing Mode" />
	<img src="https://img.shields.io/badge/RL%20Lab-Voiture%20Autonome-111827?style=for-the-badge" alt="RL Lab" />
</p>
