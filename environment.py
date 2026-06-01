import random

import numpy as np

from entropy_utils import coverage_ratio, repeated_visit_ratio, total_entropy


class GridWorld:
    """Obstacle-aware grid world used for controlled exploration experiments."""

    def __init__(self, size=10, obstacle_density=0.12, seed=None, start=(0, 0)):
        self.size = size
        self.obstacle_density = obstacle_density
        self.start = start
        self.random = random.Random(seed)
        self.seed = seed

        self.visit_map = np.zeros((size, size), dtype=float)
        self.obstacle_mask = np.zeros((size, size), dtype=bool)
        self.obstacles = []

        self._generate_obstacles()

    def _generate_obstacles(self):
        obstacle_count = int(self.size * self.size * self.obstacle_density)
        candidates = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) != self.start
        ]

        self.random.shuffle(candidates)
        self.obstacles = candidates[:obstacle_count]

        for x, y in self.obstacles:
            self.obstacle_mask[x, y] = True

    def reset_visits(self):
        self.visit_map = np.zeros((self.size, self.size), dtype=float)

    def valid_moves(self, position):
        x, y = position
        directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        moves = []

        for nx, ny in directions:
            if self.is_free((nx, ny)):
                moves.append((nx, ny))

        return moves

    def is_free(self, position):
        x, y = position
        return (
            0 <= x < self.size
            and 0 <= y < self.size
            and not self.obstacle_mask[x, y]
        )

    def visit(self, position):
        x, y = position
        self.visit_map[x, y] += 1

    def coverage(self):
        return coverage_ratio(self.visit_map, self.obstacle_mask)

    def entropy(self):
        return total_entropy(self.visit_map, self.obstacle_mask)

    def repeated_visit_ratio(self):
        return repeated_visit_ratio(self.visit_map, self.obstacle_mask)

    def clone_with_empty_visits(self):
        clone = GridWorld(
            size=self.size,
            obstacle_density=0.0,
            seed=self.seed,
            start=self.start,
        )
        clone.obstacle_density = self.obstacle_density
        clone.obstacle_mask = self.obstacle_mask.copy()
        clone.obstacles = list(self.obstacles)
        clone.reset_visits()
        return clone
