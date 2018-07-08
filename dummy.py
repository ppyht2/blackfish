import random
import numpy as np
from constants import N_ACTIONS


def dummy_decision(board):
    """ make a random dummy move
    """
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)


def dummy_net(*args):
    return np.random.rand() * 2 - 1, np.random.rand(N_ACTIONS)
