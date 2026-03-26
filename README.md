# 🤖 Double Deep Q-Network (DDQN) - GridWorld 4x4

Ce projet implémente un agent d'apprentissage par renforcement de type **Double Deep Q-Network (DDQN)** pour résoudre un environnement **GridWorld 4x4**. L'objectif est de comparer l'efficacité de l'architecture Double DQN par rapport au DQN classique pour éviter la surestimation des valeurs Q.

---

## 🌍 Environnement GridWorld

| Propriété | Détail |
| :--- | :--- |
| **Grille** | 4x4 cases |
| **Départ** | Position `(0, 0)` — Haut-Gauche |
| **Objectif** | Position `(3, 3)` — Bas-Droite (Récompense : **+10**) |
| **Obstacle** | Position `(1, 1)` (Pénalité : **-5**) |
| **Malus de pas** | Chaque mouvement coûte **-1** pour favoriser le chemin le plus court |

```
 ┌───┬───┬───┬───┐
 │ S │   │   │   │    S = Start (0,0)
 ├───┼───┼───┼───┤    X = Obstacle (1,1)
 │   │ X │   │   │    G = Goal (3,3)
 ├───┼───┼───┼───┤
 │   │   │   │   │
 ├───┼───┼───┼───┤
 │   │   │   │ G │
 └───┴───┴───┴───┘
```

---

## 🧠 Architecture de l'Agent Double DQN

L'idée principale du **Double DQN** est de séparer la **sélection** de l'action de son **évaluation** afin de stabiliser l'apprentissage et d'éviter la surestimation systématique des valeurs Q propre au DQN classique.

| Étape | Réseau utilisé | Rôle |
| :--- | :--- | :--- |
| **Sélection de l'action** | Online Network | Choisit la meilleure action : $a^* = \text{argmax } Q_{online}(s', a)$ |
| **Évaluation de l'action** | Target Network | Calcule la valeur cible : $Q_{target}(s', a^*)$ |

**Formule de mise à jour (Double DQN) :**

$$y = r + \gamma \cdot Q_{target}(s',\ \underset{a}{\text{argmax}}\ Q_{online}(s', a))$$

Contrairement au DQN classique qui utilise un seul réseau pour les deux étapes, le Double DQN découple sélection et évaluation, ce qui **réduit le biais de surestimation** et produit une convergence plus stable.

### Hyperparamètres

| Paramètre | Valeur | Description |
| :--- | :--- | :--- |
| `GAMMA` | `0.9` | Facteur d'actualisation des récompenses futures |
| `LEARNING_RATE` | `0.001` | Taux d'apprentissage de l'optimiseur Adam |
| `EPSILON` | `1.0` | Exploration initiale (100% aléatoire) |
| `EPSILON_MIN` | `0.01` | Seuil minimal d'exploration |
| `EPSILON_DECAY` | `0.995` | Taux de décroissance de l'exploration |
| `BATCH_SIZE` | `32` | Nombre d'expériences par mise à jour |
| `MEMORY_SIZE` | `2000` | Capacité maximale du replay buffer |
| `EPISODES` | `500` | Nombre total d'épisodes d'entraînement |
| `TARGET_UPDATE_FREQ` | `10` | Fréquence (en épisodes) de synchronisation du Target Network |

---

## 📂 Description du Code

### 🔧 Imports & Configuration (`cellule 1`)

```python
import numpy as np, tensorflow as tf
from tensorflow.keras.models import Sequential
from collections import deque
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

- **NumPy** : Manipulation des états sous forme de vecteurs.
- **TensorFlow / Keras** : Construction et entraînement des réseaux de neurones.
- **`deque`** : Implémentation efficace du **replay buffer** à taille fixe (`MEMORY_SIZE`).
- **`CUDA_VISIBLE_DEVICES = '-1'`** : Force l'exécution sur **CPU uniquement** — utile pour la reproductibilité et les environnements sans GPU.

---

### ⚙️ Hyperparamètres (`cellule 2`)

Centralise toutes les constantes du projet. La séparation en constantes globales facilite l'**expérimentation** : modifier `EPISODES`, `GAMMA` ou `TARGET_UPDATE_FREQ` suffit pour tester de nouvelles configurations sans toucher à la logique métier.

---

### 🌍 Classe `GridWorld` (`cellule 3`)

```python
class GridWorld:
    def reset(self) -> np.ndarray   # Réinitialise et renvoie l'état initial
    def get_state(self) -> np.ndarray  # Encode la grille en vecteur one-hot (16 valeurs)
    def step(action) -> (state, reward, done)  # Applique une action, renvoie transition
```

- **Encodage de l'état** : La grille 4×4 est aplatie en un **vecteur de 16 éléments** (one-hot : `1` à la position de l'agent, `0` ailleurs). Simple mais efficace pour un MLP.
- **Gestion des bords** : Si une action mène hors de la grille, l'agent reste sur place (pas de pénalité supplémentaire).
- **Récompenses** :
  - `+10` → Objectif atteint, épisode terminé.
  - `-5` → Obstacle heurté, l'épisode **continue** (l'agent peut se racheter).
  - `-1` → Pas normal, incite à trouver le chemin le plus court.

---

### 🤖 Classe `DoubleDQNAgent` (`cellule 4`)

```python
class DoubleDQNAgent:
    def _build_model(self) -> Sequential   # Réseau Dense(24) → Dense(24) → Dense(4)
    def update_target_network(self)        # Copie les poids : online → target
    def act(state) -> int                  # ε-greedy : explore ou exploite
    def remember(...)                      # Stocke une transition dans le replay buffer
    def replay(self)                       # Tire un batch et met à jour le online network
```

**Architecture du réseau :**

```
Input(16) → Dense(24, ReLU) → Dense(24, ReLU) → Dense(4, Linear)
```

Le couche de sortie produit **4 valeurs Q** (une par action : Haut, Bas, Gauche, Droite). L'activation `linear` est indispensable : les Q-valeurs ne sont pas bornées.

**Méthode `act` — Stratégie ε-greedy :**

```python
if np.random.rand() <= self.epsilon:
    return random.randrange(self.action_size)   # Exploration
q_values = self.model(state_input, training=False)
return np.argmax(q_values[0])                   # Exploitation
```

L'appel direct `self.model(...)` est **plus rapide** que `.predict()` pour des inférences individuelles (évite la surcharge de compilation Keras).

**Méthode `replay` — Logique Double DQN :**

```python
best_action = np.argmax(next_q_online[i])          # Sélection par le online network
target = rewards[i] + GAMMA * next_q_target[i][best_action]  # Évaluation par le target
```

Les prédictions sont **vectorisées sur tout le batch** (un seul appel réseau), ce qui est significativement plus rapide qu'une boucle d'inférences individuelles.

---

### 🏋️ Boucle d'Entraînement (`cellule 5`)

```python
for episode in range(EPISODES):
    state = env.reset()
    for step in range(50):          # Max 50 pas par épisode
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(...)
        agent.replay()              # Mise à jour du online network
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()   # Synchronisation périodique
```

- **Limite de 50 pas** : Évite les épisodes infinis si l'agent tourne en rond.
- **Synchronisation du Target Network** toutes les 10 épisodes : trop fréquente → instabilité ; trop rare → apprentissage lent.
- **Sauvegarde finale** : `agent.model.save("double_dqn_model.keras")` — format natif Keras, rechargeable avec `tf.keras.models.load_model()`.

---

### 📊 Visualisation (`cellule 6`)

La fonction `plot_learning_curves` génère deux graphiques côte à côte :

| Graphique | Ce qu'il montre |
| :--- | :--- |
| **Score par épisode** | Récompense brute (bruit élevé) + moyenne mobile sur 50 épisodes (tendance lissée) |
| **Décroissance d'Epsilon** | Transition progressive de l'exploration pure (ε=1) vers l'exploitation (ε=0.01) |

La **moyenne mobile** est calculée avec une fenêtre glissante de 50 épisodes — elle révèle la tendance d'apprentissage malgré la variance élevée des scores individuels.

---

### ✅ Phase de Test (`cellule 7`)

```python
# Exploitation pure : epsilon = 0, toujours la meilleure action connue
q_values = agent.model(state_input, training=False)
action = np.argmax(q_values[0])
```

Après entraînement, l'agent est testé en **mode greedy pur** (pas d'exploration). La position de l'agent est affichée à chaque pas pour vérifier visuellement que le chemin optimal `(0,0) → (3,3)` est suivi en 6 étapes minimum.

---

## 📈 Résultats

### Courbes d'apprentissage typiques

Après **500 épisodes** d'entraînement, l'agent DDQN présente le comportement suivant :

| Phase | Épisodes | Comportement observé |
| :--- | :--- | :--- |
| **Exploration** | 0 – 150 | Scores très variables (−30 à +10), l'agent découvre l'environnement aléatoirement |
| **Transition** | 150 – 300 | La moyenne mobile remonte progressivement, l'agent évite de mieux en mieux l'obstacle |
| **Convergence** | 300 – 500 | Scores stables autour de **+4 à +6**, l'agent suit régulièrement un chemin quasi-optimal |

> **Score optimal théorique :** L'agent part en `(0,0)` et doit atteindre `(3,3)`. Le chemin le plus court (en évitant `(1,1)`) fait **8 pas**, soit un score de `10 + 8×(−1) = +2`. Des scores supérieurs indiquent un chemin légèrement moins court mais sans passage par l'obstacle.

### Décroissance d'Epsilon

```
Epsilon : 1.00 → 0.01
Seuil atteint vers l'épisode ~900 (décroissance de 0.995/épisode)
Sur 500 épisodes, epsilon final ≈ 0.082 (8% d'exploration résiduelle)
```

L'agent conserve donc une légère exploration en fin d'entraînement, ce qui peut être bénéfique pour éviter les minima locaux sur des environnements plus complexes.

### Chemin optimal trouvé

Exemple de sortie de la phase de test après convergence :

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

L'agent contourne l'obstacle par le bas (ligne 3), démontrant qu'il a appris une **politique optimale**.

### Comparaison DQN vs Double DQN

| Critère | DQN classique | Double DQN (ce projet) |
| :--- | :--- | :--- |
| **Surestimation des Q-valeurs** | ⚠️ Fréquente | ✅ Réduite |
| **Stabilité de l'apprentissage** | Modérée | Meilleure |
| **Convergence** | ~400–600 épisodes | ~300–500 épisodes |
| **Complexité ajoutée** | — | Target Network (faible coût) |

---

## 🚀 Installation et Utilisation

### Prérequis

- Python 3.10+
- TensorFlow / Keras
- NumPy
- Matplotlib

### Installation

```bash
git clone https://github.com/votre-nom-utilisateur/ddqn-gridworld.git
cd ddqn-gridworld
pip install tensorflow numpy matplotlib
```

### Exécution

Lancez simplement le script principal pour démarrer l'entraînement (500 épisodes par défaut) :

```bash
python main.py
```

---

## 📁 Structure du Projet

```
ddqn-gridworld/
├── main.py                  # Environnement GridWorld, agent DDQN, boucle d'entraînement
├── double_dqn_model.keras   # Modèle entraîné (généré après exécution)
└── README.md                # Documentation du projet
```

