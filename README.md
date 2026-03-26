# 🤖 Double Deep Q-Network (DDQN) - GridWorld 4x4

Ce projet implémente un agent **Double DQN** pour résoudre un environnement **GridWorld 4x4**, sous forme de **notebook Jupyter**.

---

## 🌍 Environnement GridWorld

- **Grille** : 4×4 cases (état = vecteur one-hot de 16 valeurs)
- **Départ** : `(0, 0)` | **Objectif** : `(3, 3)` → Récompense `+10`, épisode terminé
- **Obstacle** : `(1, 1)` → Pénalité `-5`, épisode continue
- **Pas normal** : `-1` par mouvement
- **Actions** : 4 directions (Haut, Bas, Gauche, Droite) — si hors grille, l'agent reste sur place

---

## 🧠 Architecture Double DQN

Deux réseaux identiques :

```
Input(16) → Dense(24, ReLU) → Dense(24, ReLU) → Dense(4, Linear)
```

| Réseau | Rôle |
| :--- | :--- |
| **Online Network** | Sélectionne la meilleure action : `argmax Q_online(s', a)` |
| **Target Network** | Évalue cette action : `Q_target(s', a*)` |

Mise à jour du Target Network toutes les **10 épisodes** (`TARGET_UPDATE_FREQ`).

### Hyperparamètres

| Paramètre | Valeur |
| :--- | :--- |
| `GAMMA` | `0.9` |
| `LEARNING_RATE` | `0.001` |
| `EPSILON` initial | `1.0` |
| `EPSILON_MIN` | `0.01` |
| `EPSILON_DECAY` | `0.995` |
| `BATCH_SIZE` | `32` |
| `MEMORY_SIZE` | `2000` |
| `EPISODES` | `500` |
| `TARGET_UPDATE_FREQ` | `10` |

---

## 📓 Structure du Notebook

Le notebook est organisé en **7 cellules** à exécuter dans l'ordre :

| Cellule | Contenu |
| :--- | :--- |
| **1** | Imports et configuration (`numpy`, `tensorflow`, `matplotlib`, CPU forcé) |
| **2** | Déclaration des hyperparamètres et du dictionnaire de mouvements |
| **3** | Classe `GridWorld` — environnement, récompenses, gestion des bords |
| **4** | Classe `DoubleDQNAgent` — construction des réseaux, replay buffer, logique Double DQN |
| **5** | Boucle d'entraînement — 500 épisodes, mise à jour périodique du Target Network, sauvegarde du modèle |
| **6** | Visualisation — courbe des scores (moyenne mobile 50 épisodes) et décroissance d'epsilon |
| **7** | Test du chemin optimal — exploitation pure, affichage pas à pas de la position de l'agent |

---

## 📈 Résultats

### Courbe d'apprentissage

- **Épisodes 0–150** : Scores très variables, exploration aléatoire dominante (ε ≈ 1.0)
- **Épisodes 150–350** : La moyenne mobile sur 50 épisodes remonte, l'agent commence à éviter l'obstacle
- **Épisodes 350–500** : Convergence, scores stables, l'agent atteint régulièrement l'objectif

### Décroissance d'Epsilon

Avec `EPSILON_DECAY = 0.995` sur 500 épisodes : ε final ≈ **0.08** (~8% d'exploration résiduelle).

### Test du chemin optimal (cellule 7)

Exemple de sortie après entraînement :

```
TEST DU CHEMIN OPTIMAL
Étape 1 : Position de l'agent -> (1, 0)
Étape 2 : Position de l'agent -> (2, 0)
Étape 3 : Position de l'agent -> (3, 0)
Étape 4 : Position de l'agent -> (3, 1)
Étape 5 : Position de l'agent -> (3, 2)
Étape 6 : Position de l'agent -> (3, 3)
RÉSULTAT : Objectif atteint avec succès ! 🎉
```

L'agent contourne l'obstacle par la bordure inférieure en **6 pas** (score = `10 − 6 = +4`).

---

## 🚀 Installation et Utilisation

### Prérequis

- Python 3.10+
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook ou JupyterLab

### Installation

```bash
git clone https://github.com/votre-nom-utilisateur/ddqn-gridworld.git
cd ddqn-gridworld
pip install tensorflow numpy matplotlib notebook
```

### Exécution

```bash
jupyter notebook ddqn_gridworld.ipynb
```

Exécuter les cellules **dans l'ordre** (1 → 7). Le modèle est sauvegardé automatiquement à la fin de la cellule 5 sous `double_dqn_model.keras`.

---

## 📁 Structure du Projet

```
ddqn-gridworld/
├── ddqn_gridworld.ipynb     # Notebook principal (7 cellules)
├── double_dqn_model.keras   # Modèle sauvegardé après exécution de la cellule 5
└── README.md
```

---

*Développé dans le cadre du Master SDIA — Module SMA et IAD.*
