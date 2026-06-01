import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "outputs", ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_utils import entropy_map_from_visits


class Visualizer:
    """Visualization helpers for the single-agent demo run."""

    def __init__(self, env, output_dir="outputs"):
        self.env = env
        self.output_dir = output_dir
        self.frames = []
        self.entropy_history = []
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_entropy_map(self):
        return entropy_map_from_visits(
            self.env.visit_map,
            obstacle_mask=self.env.obstacle_mask,
        )

    def total_entropy(self, entropy_map):
        return float(np.sum(entropy_map))

    def show(self, robot, save_frame=True):
        plt.clf()

        entropy_map = self.calculate_entropy_map()
        display_map = entropy_map.copy()
        display_map[self.env.obstacle_mask] = -0.2

        plt.imshow(
            display_map,
            cmap="turbo",
            interpolation="bicubic",
            vmin=0,
            vmax=1,
        )

        plt.contour(
            display_map,
            levels=8,
            linewidths=0.5,
            colors="white",
            alpha=0.35,
        )

        path_x = [p[1] for p in robot.path]
        path_y = [p[0] for p in robot.path]
        plt.plot(path_x, path_y, color="cyan", linewidth=2, alpha=0.8)

        rx, ry = robot.position
        plt.scatter(
            ry,
            rx,
            s=260,
            color="lime",
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )

        total_h = self.total_entropy(entropy_map)
        self.entropy_history.append(total_h)

        plt.title(
            f"Adaptive Entropy Exploration | Coverage = {self.env.coverage():.2%}",
            fontsize=14,
        )
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_label("Entropy / uncertainty", fontsize=11)
        plt.tight_layout()

        if save_frame:
            frame_path = os.path.join(self.output_dir, "temp_frame.png")
            plt.savefig(frame_path)
            self._store_frame(frame_path)

        plt.pause(0.01)

    def _store_frame(self, frame_path):
        try:
            import cv2
        except ImportError:
            return

        frame = cv2.imread(frame_path)
        if frame is not None:
            self.frames.append(frame)

    def save_obstacle_map(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.env.obstacle_mask.astype(int), cmap="binary", interpolation="nearest")

        for i in range(self.env.size + 1):
            plt.axhline(i - 0.5, color="cyan", linewidth=1)
            plt.axvline(i - 0.5, color="cyan", linewidth=1)

        plt.title("Environment Obstacle Layout", fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "obstacle_map.png"), dpi=300)
        plt.close()

    def save_before_exploration(self):
        self._save_exploration_map("Before Exploration", "before_exploration.png")

    def save_after_exploration(self, robot):
        self._save_exploration_map("After Exploration", "after_exploration.png", robot)

    def _save_exploration_map(self, title, filename, robot=None):
        plt.figure(figsize=(8, 8))
        entropy_map = self.calculate_entropy_map()
        display_map = entropy_map.copy()
        display_map[self.env.obstacle_mask] = -0.2

        plt.imshow(
            display_map,
            cmap="turbo",
            interpolation="bicubic",
            vmin=0,
            vmax=1,
        )

        if robot is not None:
            path_x = [p[1] for p in robot.path]
            path_y = [p[0] for p in robot.path]
            plt.plot(path_x, path_y, color="cyan", linewidth=2)

            rx, ry = robot.position
            plt.scatter(ry, rx, s=260, color="lime", edgecolors="white")

        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def save_video(self):
        if len(self.frames) == 0:
            print("Video skipped: OpenCV is unavailable or no frames were captured.")
            return

        import cv2

        height, width, _ = self.frames[0].shape
        video = cv2.VideoWriter(
            os.path.join(self.output_dir, "exploration_video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            15,
            (width, height),
        )

        for frame in self.frames:
            video.write(frame)

        video.release()
        print("Video saved.")

    def save_entropy_graph(self):
        plt.figure(figsize=(10, 5))
        timesteps = np.arange(len(self.entropy_history))
        plt.plot(timesteps, self.entropy_history, linewidth=3, color="blue")
        plt.xlabel("Exploration Timestep", fontsize=12)
        plt.ylabel("Total Entropy", fontsize=12)
        plt.title("Total Entropy Reduction Over Exploration Steps", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "entropy_vs_time.png"), dpi=300)
        plt.close()
