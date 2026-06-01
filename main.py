import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "outputs", ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import GridWorld
from experiment_runner import run_experiments
from explorer import AdaptiveEntropyExplorer
from visualization import Visualizer


def run_demo(size, obstacle_density, steps, seed):
    env = GridWorld(size=size, obstacle_density=obstacle_density, seed=seed)
    robot = AdaptiveEntropyExplorer(env, seed=seed)
    visualizer = Visualizer(env)

    visualizer.save_obstacle_map()
    visualizer.save_before_exploration()

    plt.figure(figsize=(8, 8))

    for _ in range(steps):
        robot.step()
        visualizer.show(robot)

    visualizer.save_after_exploration(robot)
    visualizer.save_video()
    visualizer.save_entropy_graph()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run entropy-guided exploration demos or experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "experiments"],
        default="demo",
        help="demo shows one run; experiments compares all methods.",
    )
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--obstacle-density", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/experiments")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "demo":
        run_demo(
            size=args.size,
            obstacle_density=args.obstacle_density,
            steps=args.steps,
            seed=args.seed,
        )
    else:
        run_experiments(
            sizes=[10, 20, 30],
            obstacle_densities=[0.10, 0.20, 0.30],
            trials=args.trials,
            steps=args.steps,
            output_dir=args.output_dir,
        )
