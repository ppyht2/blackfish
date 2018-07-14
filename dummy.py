import random
import numpy as np
from constants import N_ACTIONS
from helper import get_basic_position_value


def dummy_decision(board):
    """ make a random dummy move
    """
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)


UNIFORM_POLICY = np.ones(N_ACTIONS) / N_ACTIONS


def dummy_policy():
    return UNIFORM_POLICY


def dummy_value():
    return np.random.rand() * 2 - 1


def dummy_net(*args):
    return dummy_value(), dummy_policy()


def dummy_material_net(env):
    value = get_basic_position_value(env)
    return value, dummy_policy()