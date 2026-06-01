# Adaptive Entropy-Guided Exploration

This project studies lightweight exploration strategies for obstacle-aware grid
environments. It compares simple baselines against an adaptive entropy-guided
count-based policy that reduces exploration pressure as map uncertainty falls.

## Research Question

Can an adaptive entropy-guided count-based policy cover obstacle-rich grid
environments more efficiently than random walk, epsilon-greedy movement, and
pure count-based exploration?

## Novelty for a Student Research Paper

The proposed method uses a dynamic exploration coefficient:

```text
score(s) = beta_t * uncertainty(s) + gamma * frontier(s) - lambda * revisit_count(s) + noise
beta_t = beta_0 * H_t / H_0
uncertainty(s) = 1 / (N(s) + 1)
```

Where:

- `N(s)` is the visit count of candidate state `s`
- `H_0` is the initial reachable-map entropy
- `H_t` is the current reachable-map entropy
- `beta_t` decays as the map becomes less uncertain
- `frontier(s)` counts unvisited valid neighbors around candidate state `s`
- Obstacles are excluded from entropy and coverage calculations

This makes the method simple, reproducible, and suitable for controlled
comparison in a student paper.

## Compared Methods

- `random_walk`: uniformly samples a valid neighboring state
- `epsilon_greedy`: usually moves to the least-visited neighbor, sometimes random
- `count_based`: uses only `1 / (N(s) + 1)` as a novelty bonus
- `adaptive_entropy`: proposed method with entropy-scaled exploration pressure

## Metrics

The experiment runner records:

- Coverage over time
- Entropy reduction over time
- Final coverage
- Repeated visit ratio
- Path length
- Steps to 80% coverage

## Repository Structure

```text
.
|-- environment.py        # Reproducible obstacle-aware grid world
|-- entropy_utils.py      # Entropy, coverage, revisit metrics
|-- explorer.py           # Baseline and proposed exploration agents
|-- experiment_runner.py  # Multi-trial experiments and comparison plots
|-- visualization.py      # Single-run visualization helpers
|-- main.py               # CLI entry point
|-- requirements.txt
|-- outputs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Visual Demo

```bash
python main.py --mode demo --size 10 --obstacle-density 0.12 --steps 200
```

Outputs are written to `outputs/`.

## Run Research Experiments

For a quick smoke test:

```bash
python main.py --mode experiments --steps 100 --trials 3
```

For the full student-paper experiment:

```bash
python main.py --mode experiments --steps 500 --trials 50
```

Experiment outputs are written to `outputs/experiments/`:

- `timeseries_metrics.csv`
- `summary_metrics.csv`
- `coverage_vs_time.png`
- `entropy_reduction_vs_time.png`
- `final_coverage_by_method.png`
- `revisit_ratio_by_method.png`

## Suggested Paper Claim

This work introduces an adaptive entropy-guided count-based exploration method
for grid-world navigation. The method dynamically reduces intrinsic exploration
pressure as reachable-map uncertainty decreases and is evaluated against common
exploration baselines across randomized obstacle densities and grid sizes.

## Related Work to Cite

- Shannon, C. E. A Mathematical Theory of Communication.
- Bellemare et al. Unifying Count-Based Exploration and Intrinsic Motivation.
- Tang et al. #Exploration: A Study of Count-Based Exploration for Deep RL.
- Houthooft et al. VIME: Variational Information Maximizing Exploration.
- Pathak et al. Curiosity-driven Exploration by Self-supervised Prediction.
- Burda et al. Exploration by Random Network Distillation.
- Hazan et al. Provably Efficient Maximum Entropy Exploration.
- Seo et al. State Entropy Maximization with Random Encoders.
- Yamauchi. A Frontier-Based Approach for Autonomous Exploration.
