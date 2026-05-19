# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class Visualizer:

    def __init__(self, env):

        self.env = env

        # Store frames for video
        self.frames = []

        # Store entropy values over time
        self.entropy_history = []

        # Create output folder
        os.makedirs("outputs", exist_ok=True)

    # =========================
    # Calculate Entropy Map
    # =========================
    def calculate_entropy_map(self):

        visits = self.env.visit_map

        # Exponential uncertainty decay
        entropy_map = np.exp(-0.7 * visits)

        # Gamma correction for strong contrast
        entropy_map = entropy_map ** 2.2

        return entropy_map

    # =========================
    # Total Entropy
    # =========================
    def total_entropy(self, entropy_map):

        return np.sum(entropy_map)

    # =========================
    # Main Visualization
    # =========================
    def show(self, robot, save_frame=True):

        plt.clf()

        entropy_map = self.calculate_entropy_map()

        # Add obstacles
        for x, y in self.env.obstacles:
            entropy_map[x][y] = -0.2

        # Heatmap
        plt.imshow(
            entropy_map,
            cmap='turbo',
            interpolation='bicubic',
            vmin=0,
            vmax=1
        )

        # Contour lines
        plt.contour(
            entropy_map,
            levels=8,
            linewidths=0.5,
            colors='white',
            alpha=0.35
        )

        # Robot path
        path_x = [p[1] for p in robot.path]
        path_y = [p[0] for p in robot.path]

        plt.plot(
            path_x,
            path_y,
            color='cyan',
            linewidth=2,
            alpha=0.8
        )

        # Current robot position
        rx, ry = robot.position

        plt.scatter(
            ry,
            rx,
            s=260,
            color='lime',
            edgecolors='white',
            linewidths=1.5,
            zorder=5
        )

        # Calculate total entropy
        total_H = self.total_entropy(entropy_map)

        # Save entropy history
        self.entropy_history.append(total_H)

        # Title
        plt.title(
            f"Entropy-Driven Exploration | Total Entropy = {total_H:.2f}",
            fontsize=14
        )

        plt.xticks([])
        plt.yticks([])

        # Colorbar
        cbar = plt.colorbar(
            fraction=0.046,
            pad=0.04
        )

        cbar.set_label(
            "Entropy / Uncertainty",
            fontsize=11
        )

        plt.tight_layout()

        # Save frame for video
        if save_frame:

            frame_path = "outputs/temp_frame.png"

            plt.savefig(frame_path)

            frame = cv2.imread(frame_path)

            self.frames.append(frame)

        plt.pause(0.05)

    # =========================
    # Save Obstacle Map
    # =========================
    def save_obstacle_map(self):

        plt.figure(figsize=(6, 6))

        obstacle_map = np.zeros(
            (self.env.size, self.env.size)
        )

        # Mark obstacles
        for x, y in self.env.obstacles:
            obstacle_map[x][y] = 1

        plt.imshow(
            obstacle_map,
            cmap='binary',
            interpolation='nearest'
        )

        # Grid overlay
        for i in range(self.env.size + 1):

            plt.axhline(
                i - 0.5,
                color='cyan',
                linewidth=1
            )

            plt.axvline(
                i - 0.5,
                color='cyan',
                linewidth=1
            )

        plt.title(
            "Environment Obstacle Layout",
            fontsize=14
        )

        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        plt.savefig(
            "outputs/obstacle_map.png",
            dpi=300
        )

        plt.close()

    # =========================
    # Save Before Exploration
    # =========================
    def save_before_exploration(self):

        plt.figure(figsize=(8, 8))

        entropy_map = self.calculate_entropy_map()

        for x, y in self.env.obstacles:
            entropy_map[x][y] = -0.2

        plt.imshow(
            entropy_map,
            cmap='turbo',
            interpolation='bicubic',
            vmin=0,
            vmax=1
        )

        plt.title("Before Exploration")

        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        plt.savefig(
            "outputs/before_exploration.png",
            dpi=300
        )

        plt.close()

    # =========================
    # Save After Exploration
    # =========================
    def save_after_exploration(self, robot):

        plt.figure(figsize=(8, 8))

        entropy_map = self.calculate_entropy_map()

        for x, y in self.env.obstacles:
            entropy_map[x][y] = -0.2

        plt.imshow(
            entropy_map,
            cmap='turbo',
            interpolation='bicubic',
            vmin=0,
            vmax=1
        )

        # Robot path
        path_x = [p[1] for p in robot.path]
        path_y = [p[0] for p in robot.path]

        plt.plot(
            path_x,
            path_y,
            color='cyan',
            linewidth=2
        )

        # Final robot position
        rx, ry = robot.position

        plt.scatter(
            ry,
            rx,
            s=260,
            color='lime',
            edgecolors='white'
        )

        plt.title("After Exploration")

        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        plt.savefig(
            "outputs/after_exploration.png",
            dpi=300
        )

        plt.close()

    # =========================
    # Save Exploration Video
    # =========================
    def save_video(self):

        if len(self.frames) == 0:
            return

        height, width, _ = self.frames[0].shape

        video = cv2.VideoWriter(
            "outputs/exploration_video.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            15,
            (width, height)
        )

        for frame in self.frames:
            video.write(frame)

        video.release()

        print("Video saved!")

    # =========================
    # Save Entropy vs Time Graph
    # =========================
    def save_entropy_graph(self):

        plt.figure(figsize=(10, 5))

        # X-axis → timesteps
        timesteps = np.arange(
            len(self.entropy_history)
        )

        # Y-axis → total entropy
        plt.plot(
            timesteps,
            self.entropy_history,
            linewidth=3,
            color='blue'
        )

        # Labels
        plt.xlabel(
            "Exploration Timestep",
            fontsize=12
        )

        plt.ylabel(
            "Total Entropy",
            fontsize=12
        )

        # Title
        plt.title(
            "Total Entropy Reduction Over Exploration Steps",
            fontsize=14
        )

        # Grid
        plt.grid(alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save graph
        plt.savefig(
            "outputs/entropy_vs_time.png",
            dpi=300
        )

        plt.close()