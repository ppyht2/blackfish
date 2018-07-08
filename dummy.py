import random


def dummy_decision(board):
    """ make a random dummy move
    """
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)
