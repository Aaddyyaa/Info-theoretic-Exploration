import numpy as np


def uncertainty_from_visits(visit_count):
    """Count-based uncertainty bonus for a single state."""
    return 1.0 / (visit_count + 1.0)


def entropy_map_from_visits(visit_map, obstacle_mask=None, decay=0.7):
    """Return an uncertainty/entropy proxy map from state visit counts."""
    entropy_map = np.exp(-decay * visit_map.astype(float))

    if obstacle_mask is not None:
        entropy_map = entropy_map.copy()
        entropy_map[obstacle_mask] = 0.0

    return entropy_map


def total_entropy(visit_map, obstacle_mask=None, decay=0.7):
    """Total reachable-map uncertainty."""
    return float(np.sum(entropy_map_from_visits(visit_map, obstacle_mask, decay)))


def coverage_ratio(visit_map, obstacle_mask=None):
    """Fraction of non-obstacle states visited at least once."""
    if obstacle_mask is None:
        reachable = np.ones_like(visit_map, dtype=bool)
    else:
        reachable = ~obstacle_mask

    reachable_count = int(np.sum(reachable))
    if reachable_count == 0:
        return 0.0

    visited = (visit_map > 0) & reachable
    return float(np.sum(visited) / reachable_count)


def repeated_visit_ratio(visit_map, obstacle_mask=None):
    """How much of the trajectory revisits already explored states."""
    if obstacle_mask is None:
        reachable = np.ones_like(visit_map, dtype=bool)
    else:
        reachable = ~obstacle_mask

    visits = visit_map[reachable]
    total_visits = float(np.sum(visits))
    unique_visits = float(np.sum(visits > 0))

    if total_visits == 0:
        return 0.0

    return float((total_visits - unique_visits) / total_visits)
