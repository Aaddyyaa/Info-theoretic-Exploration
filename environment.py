import numpy as np
import random

class GridWorld:

    def __init__(self, size=10):

        self.size = size

        self.visit_map = np.zeros((size, size))

        self.obstacles = []

        # Random obstacles
        for _ in range(12):

            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)

            if (x, y) != (0, 0):
                self.obstacles.append((x, y))

    def valid_moves(self, position):

        x, y = position

        directions = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1)
        ]

        moves = []

        for nx, ny in directions:

            if (
                0 <= nx < self.size
                and 0 <= ny < self.size
                and (nx, ny) not in self.obstacles
            ):
                moves.append((nx, ny))

        return moves

    def visit(self, position):

        x, y = position

        self.visit_map[x][y] += 1