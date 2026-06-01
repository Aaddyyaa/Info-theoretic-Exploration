import random

from entropy_utils import uncertainty_from_visits


class BaseExplorer:
    """Shared movement and metric tracking for all exploration policies."""

    name = "base"

    def __init__(self, env, seed=None):
        self.env = env
        self.random = random.Random(seed)
        self.position = env.start
        self.previous_position = None
        self.path = [self.position]
        self.initial_entropy = max(self.env.entropy(), 1e-9)

        self.env.visit(self.position)

    def choose_action(self):
        raise NotImplementedError

    def step(self):
        move = self.choose_action()
        if move is None:
            return self.position

        self.previous_position = self.position
        self.position = move
        self.path.append(self.position)
        self.env.visit(self.position)
        return self.position

    def metrics(self, step):
        return {
            "step": step,
            "coverage": self.env.coverage(),
            "entropy": self.env.entropy(),
            "entropy_reduction": 1.0 - (self.env.entropy() / self.initial_entropy),
            "repeated_visit_ratio": self.env.repeated_visit_ratio(),
            "path_length": len(self.path) - 1,
        }


class RandomWalkExplorer(BaseExplorer):
    name = "random_walk"

    def choose_action(self):
        moves = self.env.valid_moves(self.position)
        if not moves:
            return None
        return self.random.choice(moves)


class EpsilonGreedyExplorer(BaseExplorer):
    name = "epsilon_greedy"

    def __init__(self, env, epsilon=0.20, seed=None):
        super().__init__(env, seed=seed)
        self.epsilon = epsilon

    def choose_action(self):
        moves = self.env.valid_moves(self.position)
        if not moves:
            return None

        if self.random.random() < self.epsilon:
            return self.random.choice(moves)

        return min(
            moves,
            key=lambda pos: self.env.visit_map[pos[0], pos[1]],
        )


class CountBasedExplorer(BaseExplorer):
    name = "count_based"

    def choose_action(self):
        moves = self.env.valid_moves(self.position)
        if not moves:
            return None

        best_score = -float("inf")
        best_move = None

        for pos in moves:
            visits = self.env.visit_map[pos[0], pos[1]]
            score = uncertainty_from_visits(visits)
            score += self.random.uniform(0.0, 0.02)

            if score > best_score:
                best_score = score
                best_move = pos

        return best_move


class AdaptiveEntropyExplorer(BaseExplorer):
    name = "adaptive_entropy"

    def __init__(self, env, beta0=1.0, distance_cost=0.03, frontier_weight=0.05, seed=None):
        super().__init__(env, seed=seed)
        self.beta0 = beta0
        self.distance_cost = distance_cost
        self.frontier_weight = frontier_weight

    def current_beta(self):
        return self.beta0 * (self.env.entropy() / self.initial_entropy)

    def choose_action(self):
        moves = self.env.valid_moves(self.position)
        if not moves:
            return None

        beta_t = self.current_beta()
        best_score = -float("inf")
        best_move = None

        for pos in moves:
            visits = self.env.visit_map[pos[0], pos[1]]
            uncertainty = uncertainty_from_visits(visits)
            frontier_bonus = self._frontier_bonus(pos)
            revisit_penalty = self.distance_cost * visits
            random_bonus = self.random.uniform(0.0, 0.02)
            score = (
                (beta_t * uncertainty)
                + (self.frontier_weight * frontier_bonus)
                - revisit_penalty
                + random_bonus
            )

            if score > best_score:
                best_score = score
                best_move = pos

        return best_move

    def _frontier_bonus(self, position):
        unvisited_neighbors = 0

        for nx, ny in self.env.valid_moves(position):
            if self.env.visit_map[nx, ny] == 0:
                unvisited_neighbors += 1

        return unvisited_neighbors


# Backward-compatible alias for the original demo name.
EntropyExplorer = AdaptiveEntropyExplorer


EXPLORERS = {
    RandomWalkExplorer.name: RandomWalkExplorer,
    EpsilonGreedyExplorer.name: EpsilonGreedyExplorer,
    CountBasedExplorer.name: CountBasedExplorer,
    AdaptiveEntropyExplorer.name: AdaptiveEntropyExplorer,
}
