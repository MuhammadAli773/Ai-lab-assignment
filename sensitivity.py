import random
import numpy as np
import matplotlib.pyplot as plt
from uav_ga import Environment, GeneticAlgorithm

env = Environment()
RUNS = 3
GENS = 40
tests = {
    "Population size": ("pop_size", [10, 40, 80]),
    "Mutation rate": ("mutation_rate", [0.02, 0.15, 0.40]),
    "Tournament size": ("tournament", [2, 5, 10]),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, (param, values)) in zip(axes, tests.items()):
    for v in values:
        curves = []
        for r in range(RUNS):
            random.seed(r)
            np.random.seed(r)
            ga = GeneticAlgorithm(env, generations=GENS, **{param: v})
            _, best_hist, _, _ = ga.run(verbose=False)
            curves.append(best_hist)
        mean_curve = np.mean(curves, axis=0)
        ax.plot(mean_curve, label=f"{param} = {v}")
        print(f"{title:16s} = {v:5}  ->  final best fitness (avg of {RUNS} runs) = {mean_curve[-1]:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sensitivity.png", dpi=150)
print("Saved sensitivity.png")
