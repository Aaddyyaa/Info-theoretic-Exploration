import matplotlib.pyplot as plt

from environment import GridWorld
from explorer import EntropyExplorer
from visualization import Visualizer

# Environment
env = GridWorld(size=10)

# Robot
robot = EntropyExplorer(env)

# Visualization
visualizer = Visualizer(env)

# Initial visit
env.visit((0, 0))

# Save initial maps
visualizer.save_obstacle_map()

visualizer.save_before_exploration()

# Main figure
plt.figure(figsize=(8, 8))

# Exploration loop
for _ in range(200):

    robot.step()

    visualizer.show(robot)

# Save final map
visualizer.save_after_exploration(robot)

# Save exploration video
visualizer.save_video()
visualizer.save_entropy_graph()
plt.show()