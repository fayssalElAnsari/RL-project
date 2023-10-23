## PRESENTATION PROJET

Dans ce projet nous allons appliquer les algos suivants sur forzenLake:

- [ ] Q-Learning
- [ ] Double Q-Learning
- [ ] Delayed Q-learning
- [ ] DQN learning

Pour comparer les different algorithms, on peut utiliser les metriques suivants:

- Longueur d’episode: c’est a dire en combien d’etapes l’agent est capable a terminer l’episode (en gagnant)
- Taux de reussite: On peut - apres avoir appris le model - lancer plusieurs episodes de test, et calculer le taux de reussite, par exemple sur 100 episodes de test on a 75 episodes reussis, donc l’agent du model a un taux de reussite de 75%.
- Vitesse d’apprentissage: Ceci se rapporte à la rapidité avec laquelle l'algorithme converge vers une bonne stratégie (ou politique). On peut mesurer le nombre d'épisodes nécessaires pour atteindre un certain seuil de performance ou le taux d'amélioration dans les premiers épisodes.
- Efficacité Computationnelle: On peut Mesurer le temps réel (en secondes ou minutes) nécessaire pour entraîner l'algorithme. En prennat en compte des facteurs tels que la complexité de l'algorithme, la taille du réseau neuronal dans le DQN, etc.

Pour chacun des algorithmes etudies nous allons comparer avec et sans le parametre `is_slippery=True` et avec et sans la récompense négative, donc au totale il y aura 4 cas differents juste pour les parametre du jeu. ET 4 algorithmes differents donc au total il y aura **16 modeles** a entrainer et comparer.

### TODO

- [ ] ajout de metriques pour chaque algo
- [ ] comparaison et conclusion de la performance de chaque algo

## Voir Aussi

- https://gymnasium.farama.org/environments/toy_text/frozen_lake/#frozen-lake

### References

- https://github.com/FareedKhan-dev/Reinforcement-Learning-on-Frozen-Lake-v1-openAI-gym
