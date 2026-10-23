import random
import numpy as np
import matplotlib.pyplot as plt

GRID = 50
N_UAV = 3
N_WP = 6
BASE = (2, 2)
MAX_RANGE = 150

POP_SIZE = 40
GENERATIONS = 80
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3
ELITE = 2

W_COVER, W_DETECT = 0.35, 0.25
W_ENERGY, W_TIME, W_HAZARD = 0.15, 0.10, 0.15
PENALTY = 5.0


class Environment:
    def __init__(self):
        Y, X = np.mgrid[0:GRID, 0:GRID]
        self.X, self.Y = X, Y

        self.priority = np.zeros((GRID, GRID))
        self.priority[(X - 12) ** 2 + (Y - 38) ** 2 <= 7 ** 2] = 3
        self.priority[(X - 35) ** 2 + (Y - 40) ** 2 <= 6 ** 2] = 2
        self.priority[(X - 40) ** 2 + (Y - 12) ** 2 <= 6 ** 2] = 3

        self.hazard = np.zeros((GRID, GRID))
        self.hazard[(X - 35) ** 2 + (Y - 40) ** 2 <= 4 ** 2] = 1
        self.hazard[(X - 12) ** 2 + (Y - 22) ** 2 <= 5 ** 2] = 1

        self.nofly = np.zeros((GRID, GRID))
        self.nofly[10:18, 20:28] = 1
        self.nofly[30:38, 22:30] = 1

        rng = random.Random(1)
        cells = [(x, y) for x in range(GRID) for y in range(GRID) if self.priority[y, x] > 0]
        self.targets = rng.sample(cells, 15)

    def is_free(self, x, y):
        return self.nofly[int(round(y)), int(round(x))] == 0


class Chromosome:
    def __init__(self, genes=None):
        self.genes = genes if genes is not None else self.random_genes()
        self.fitness = None

    @staticmethod
    def random_genes():
        g = np.zeros((N_UAV, N_WP, 4))
        g[:, :, 0] = np.random.uniform(0, GRID - 1, (N_UAV, N_WP))
        g[:, :, 1] = np.random.uniform(0, GRID - 1, (N_UAV, N_WP))
        g[:, :, 2] = np.random.uniform(1, 3, (N_UAV, N_WP))
        g[:, :, 3] = np.random.uniform(0, 1, (N_UAV, N_WP))
        return g

    def copy(self):
        c = Chromosome(self.genes.copy())
        c.fitness = self.fitness
        return c


def route_length(route):
    return sum(np.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
               for i in range(len(route) - 1))


def repair(chrom, env):
    g = chrom.genes
    for u in range(N_UAV):
        for w in range(N_WP):
            while not env.is_free(g[u, w, 0], g[u, w, 1]):
                g[u, w, 0] = random.uniform(0, GRID - 1)
                g[u, w, 1] = random.uniform(0, GRID - 1)
        route = [BASE] + [(g[u, w, 0], g[u, w, 1]) for w in range(N_WP)] + [BASE]
        length = route_length(route)
        if length > MAX_RANGE:
            f = 0.97 * MAX_RANGE / length
            g[u, :, 0] = BASE[0] + (g[u, :, 0] - BASE[0]) * f
            g[u, :, 1] = BASE[1] + (g[u, :, 1] - BASE[1]) * f


def evaluate(chrom, env, return_details=False):
    repair(chrom, env)
    g = chrom.genes
    covered = np.zeros((GRID, GRID), dtype=bool)
    energy, hazard_pts, nofly_pts, total_pts = 0.0, 0, 0, 0
    drone_times, paths = [], []

    for u in range(N_UAV):
        route = [BASE] + [(g[u, w, 0], g[u, w, 1]) for w in range(N_WP)] + [BASE]
        paths.append(route)
        t = 0.0
        for i in range(len(route) - 1):
            (x1, y1), (x2, y2) = route[i], route[i + 1]
            w = min(i, N_WP - 1)
            speed, sensor = g[u, w, 2], g[u, w, 3]
            dist = np.hypot(x2 - x1, y2 - y1)
            t += dist / speed
            energy += dist * (0.5 + 0.2 * speed ** 2 + 0.5 * sensor)
            radius = 1.5 + 2.5 * sensor

            steps = max(1, int(dist / 2))
            for s in range(steps + 1):
                px = x1 + (x2 - x1) * s / steps
                py = y1 + (y2 - y1) * s / steps
                covered |= (env.X - px) ** 2 + (env.Y - py) ** 2 <= radius ** 2
                cx = int(round(min(max(px, 0), GRID - 1)))
                cy = int(round(min(max(py, 0), GRID - 1)))
                hazard_pts += env.hazard[cy, cx]
                nofly_pts += env.nofly[cy, cx]
                total_pts += 1
        drone_times.append(t)

    coverage = (env.priority * covered).sum() / env.priority.sum()
    detection = np.mean([covered[y, x] for (x, y) in env.targets])
    energy_n = energy / (N_UAV * MAX_RANGE * 2)
    time_n = max(drone_times) / MAX_RANGE
    hazard_n = hazard_pts / total_pts
    penalty = PENALTY * nofly_pts / total_pts

    fit = (W_COVER * coverage + W_DETECT * detection
           - W_ENERGY * energy_n - W_TIME * time_n - W_HAZARD * hazard_n - penalty)
    chrom.fitness = fit
    if return_details:
        return dict(paths=paths, covered=covered, coverage=coverage, detection=detection,
                    energy=energy, time=max(drone_times), hazard=hazard_n, penalty=penalty)
    return fit


def tournament_selection(pop, k):
    return max(random.sample(pop, k), key=lambda c: c.fitness)


def crossover(p1, p2):
    c1, c2 = p1.genes.copy(), p2.genes.copy()
    for u in range(N_UAV):
        if random.random() < 0.5:
            a, b = sorted(random.sample(range(N_WP + 1), 2))
            c1[u, a:b], c2[u, a:b] = p2.genes[u, a:b].copy(), p1.genes[u, a:b].copy()
        else:
            alpha = random.random()
            c1[u] = alpha * p1.genes[u] + (1 - alpha) * p2.genes[u]
            c2[u] = (1 - alpha) * p1.genes[u] + alpha * p2.genes[u]
    return Chromosome(c1), Chromosome(c2)


def mutate(chrom, rate):
    g = chrom.genes
    for u in range(N_UAV):
        for w in range(N_WP):
            if random.random() < rate:
                g[u, w, 0] += random.gauss(0, 4)
                g[u, w, 1] += random.gauss(0, 4)
            if random.random() < rate * 0.2:
                g[u, w, 0] = random.uniform(0, GRID - 1)
                g[u, w, 1] = random.uniform(0, GRID - 1)
            if random.random() < rate:
                g[u, w, 2] += random.gauss(0, 0.3)
                g[u, w, 3] += random.gauss(0, 0.1)
    g[:, :, 0:2] = np.clip(g[:, :, 0:2], 0, GRID - 1)
    g[:, :, 2] = np.clip(g[:, :, 2], 1, 3)
    g[:, :, 3] = np.clip(g[:, :, 3], 0, 1)


class GeneticAlgorithm:
    def __init__(self, env, pop_size=POP_SIZE, generations=GENERATIONS,
                 mutation_rate=MUTATION_RATE, tournament=TOURNAMENT_SIZE):
        self.env, self.pop_size, self.generations = env, pop_size, generations
        self.mutation_rate, self.tournament = mutation_rate, tournament

    def run(self, verbose=True):
        pop = [Chromosome() for _ in range(self.pop_size)]
        best_hist, avg_hist, snapshots = [], [], {}
        best_ever = None
        for gen in range(self.generations):
            for c in pop:
                evaluate(c, self.env)
            pop.sort(key=lambda c: c.fitness, reverse=True)
            if best_ever is None or pop[0].fitness > best_ever.fitness:
                best_ever = pop[0].copy()
            best_hist.append(pop[0].fitness)
            avg_hist.append(np.mean([c.fitness for c in pop]))
            if gen in (0, self.generations // 2):
                snapshots[gen] = evaluate(pop[0].copy(), self.env, True)
            if verbose and gen % 10 == 0:
                print(f"Generation {gen:3d}  best = {best_hist[-1]:.4f}  average = {avg_hist[-1]:.4f}")

            new_pop = [c.copy() for c in pop[:ELITE]]
            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(pop, self.tournament)
                p2 = tournament_selection(pop, self.tournament)
                if random.random() < CROSSOVER_RATE:
                    c1, c2 = crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                mutate(c1, self.mutation_rate)
                mutate(c2, self.mutation_rate)
                new_pop += [c1, c2]
            pop = new_pop[:self.pop_size]
        snapshots[self.generations - 1] = evaluate(best_ever.copy(), self.env, True)
        return best_ever, best_hist, avg_hist, snapshots


def random_search(env, n_evals):
    best, best_hist = None, []
    for i in range(n_evals):
        c = Chromosome()
        evaluate(c, env)
        if best is None or c.fitness > best.fitness:
            best = c.copy()
        if (i + 1) % POP_SIZE == 0:
            best_hist.append(best.fitness)
    return best, best_hist


COLORS = ["blue", "green", "red"]


def draw_map(ax, env, details, title):
    ax.imshow(env.priority, origin="lower", cmap="YlOrRd", alpha=0.6)
    ax.imshow(np.ma.masked_where(env.hazard == 0, env.hazard), origin="lower", cmap="Purples", alpha=0.5)
    ax.imshow(np.ma.masked_where(env.nofly == 0, env.nofly), origin="lower", cmap="gray", alpha=0.9)
    ax.scatter([t[0] for t in env.targets], [t[1] for t in env.targets], c="black", marker="x", s=15)
    ax.plot(*BASE, "*", color="gold", markersize=14, markeredgecolor="k")
    for u, route in enumerate(details["paths"]):
        ax.plot([p[0] for p in route], [p[1] for p in route], "-o", color=COLORS[u], ms=3, label=f"UAV {u+1}")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, GRID - 1)
    ax.set_ylim(0, GRID - 1)


def plot_results(env, best_hist, avg_hist, rs_hist, snapshots):
    plt.figure(figsize=(7, 4.5))
    plt.plot(best_hist, label="GA best")
    plt.plot(avg_hist, label="GA average")
    plt.plot(rs_hist, "--", label="Random search best")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.close()

    gens = sorted(snapshots)
    fig, axes = plt.subplots(2, len(gens), figsize=(5 * len(gens), 9))
    for i, g in enumerate(gens):
        d = snapshots[g]
        draw_map(axes[0, i], env, d, f"Generation {g}: coverage = {d['coverage']:.2f}")
        axes[1, i].imshow(d["covered"], origin="lower", cmap="Greens")
        axes[1, i].set_title(f"Coverage heatmap - generation {g}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("paths_evolution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    env = Environment()

    print("Running Genetic Algorithm...")
    ga = GeneticAlgorithm(env)
    best, best_hist, avg_hist, snapshots = ga.run()

    print("\nRunning Random Search (same number of evaluations)...")
    rs_best, rs_hist = random_search(env, POP_SIZE * GENERATIONS)

    plot_results(env, best_hist, avg_hist, rs_hist, snapshots)

    d = snapshots[GENERATIONS - 1]
    print("\n===== FINAL RESULT (Genetic Algorithm) =====")
    print(f"Fitness   : {best.fitness:.4f}   (random search: {rs_best.fitness:.4f})")
    print(f"Coverage  : {d['coverage']:.2f}")
    print(f"Detection : {d['detection']:.2f}")
    print(f"Energy    : {d['energy']:.0f}")
    print(f"Time      : {d['time']:.1f}")
    print(f"Hazard    : {d['hazard']:.3f}")
    print("Plots saved: convergence.png, paths_evolution.png")
