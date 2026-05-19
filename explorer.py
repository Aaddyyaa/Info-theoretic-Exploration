import random

class EntropyExplorer:

    def __init__(self, env):

        self.env = env

        self.position = (0, 0)

        self.previous_position = None

        self.path = [self.position]

    def choose_action(self):

        moves = self.env.valid_moves(self.position)

        best_score = -999

        best_move = None

        for pos in moves:

            x, y = pos

            visits = self.env.visit_map[x][y]

            curiosity = 1 / (visits + 1)

            random_bonus = random.uniform(0, 0.15)

            score = curiosity + random_bonus

            if score > best_score:

                best_score = score

                best_move = pos

        return best_move

    def step(self):

        self.previous_position = self.position

        self.position = self.choose_action()

        self.path.append(self.position)

        self.env.visit(self.position)