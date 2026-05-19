import numpy as np

def calculate_curiosity(visit_count):

    return 1 / (visit_count + 1)