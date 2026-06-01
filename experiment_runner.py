import argparse
import csv
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "outputs", ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment import GridWorld
from explorer import EXPLORERS


def run_single_trial(agent_name, size, obstacle_density, steps, seed):
    base_env = GridWorld(
        size=size,
        obstacle_density=obstacle_density,
        seed=seed,
    )
    env = base_env.clone_with_empty_visits()
    agent = EXPLORERS[agent_name](env, seed=seed)

    rows = [agent.metrics(step=0)]

    for step in range(1, steps + 1):
        agent.step()
        rows.append(agent.metrics(step=step))

    final_metrics = dict(rows[-1])
    final_metrics["steps_to_80_coverage"] = steps_to_coverage(rows, target=0.80)
    return rows, final_metrics


def steps_to_coverage(rows, target):
    for row in rows:
        if row["coverage"] >= target:
            return row["step"]
    return None


def run_experiments(
    sizes,
    obstacle_densities,
    trials,
    steps,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    timeseries_rows = []
    summary_rows = []

    for size in sizes:
        for density in obstacle_densities:
            for trial in range(trials):
                seed = 10_000 * size + int(density * 1_000) + trial

                for agent_name in EXPLORERS:
                    rows, final_metrics = run_single_trial(
                        agent_name=agent_name,
                        size=size,
                        obstacle_density=density,
                        steps=steps,
                        seed=seed,
                    )

                    for row in rows:
                        row.update(
                            {
                                "agent": agent_name,
                                "size": size,
                                "obstacle_density": density,
                                "trial": trial,
                                "seed": seed,
                            }
                        )
                        timeseries_rows.append(row)

                    final_metrics.update(
                        {
                            "agent": agent_name,
                            "size": size,
                            "obstacle_density": density,
                            "trial": trial,
                            "seed": seed,
                        }
                    )
                    summary_rows.append(final_metrics)

    write_csv(
        os.path.join(output_dir, "timeseries_metrics.csv"),
        timeseries_rows,
    )
    write_csv(
        os.path.join(output_dir, "summary_metrics.csv"),
        summary_rows,
    )
    plot_comparisons(timeseries_rows, summary_rows, output_dir)

    return timeseries_rows, summary_rows


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = sorted({field for row in rows for field in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_timeseries(rows, metric):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["agent"],
            row["size"],
            row["obstacle_density"],
            row["step"],
        )
        grouped[key].append(row[metric])

    aggregated = []
    for (agent, size, density, step), values in grouped.items():
        aggregated.append(
            {
                "agent": agent,
                "size": size,
                "obstacle_density": density,
                "step": step,
                metric: float(np.mean(values)),
            }
        )

    return aggregated


def plot_comparisons(timeseries_rows, summary_rows, output_dir):
    plot_timeseries_metric(
        timeseries_rows,
        metric="coverage",
        ylabel="Mean coverage",
        output_path=os.path.join(output_dir, "coverage_vs_time.png"),
    )
    plot_timeseries_metric(
        timeseries_rows,
        metric="entropy_reduction",
        ylabel="Mean entropy reduction",
        output_path=os.path.join(output_dir, "entropy_reduction_vs_time.png"),
    )
    plot_summary_metric(
        summary_rows,
        metric="coverage",
        ylabel="Final coverage",
        output_path=os.path.join(output_dir, "final_coverage_by_method.png"),
    )
    plot_summary_metric(
        summary_rows,
        metric="repeated_visit_ratio",
        ylabel="Repeated visit ratio",
        output_path=os.path.join(output_dir, "revisit_ratio_by_method.png"),
    )


def plot_timeseries_metric(rows, metric, ylabel, output_path):
    aggregated = aggregate_timeseries(rows, metric)
    scenarios = sorted({(row["size"], row["obstacle_density"]) for row in aggregated})
    agents = list(EXPLORERS.keys())

    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(10, max(4, 3.5 * len(scenarios))),
        squeeze=False,
    )

    for ax, (size, density) in zip(axes[:, 0], scenarios):
        for agent in agents:
            series = sorted(
                [
                    row
                    for row in aggregated
                    if row["agent"] == agent
                    and row["size"] == size
                    and row["obstacle_density"] == density
                ],
                key=lambda row: row["step"],
            )
            if not series:
                continue

            ax.plot(
                [row["step"] for row in series],
                [row[metric] for row in series],
                label=agent,
                linewidth=2,
            )

        ax.set_title(f"{size}x{size}, obstacle density={density:.2f}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_summary_metric(rows, metric, ylabel, output_path):
    agents = list(EXPLORERS.keys())
    values = [
        np.mean([row[metric] for row in rows if row["agent"] == agent])
        for agent in agents
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(agents, values, color=["#5b8ff9", "#61d4b3", "#f6bd16", "#e8684a"])
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run obstacle-aware exploration baselines."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument(
        "--obstacle-densities",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30],
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/experiments")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        sizes=args.sizes,
        obstacle_densities=args.obstacle_densities,
        trials=args.trials,
        steps=args.steps,
        output_dir=args.output_dir,
    )
